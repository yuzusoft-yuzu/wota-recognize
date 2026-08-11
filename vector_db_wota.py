"""
Wota艺 向量数据库模块
=====================
基于 Faiss 实现技术向量的入库、持久化与近似最近邻（ANN）检索。

设计思路：
  - 每段视频提取一个特征向量（目前 12 维光流特征，后续可扩展为 A路骨骼+B路光流→1024维）
  - 入库时自动构建/更新 Faiss 索引
  - 搜索时返回 Top-K 匹配技术名称 + 置信度百分比
  - 索引与元数据支持 JSON/Pickle 持久化

依赖：
  pip install faiss-cpu   # CPU 版（GPU 版: faiss-gpu）
"""

from __future__ import annotations

import os
import json
import pickle
import shutil
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
import warnings

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    warnings.warn("faiss 未安装，请执行: pip install faiss-cpu")


# ===================================================================
# 1. 数据结构
# ===================================================================
@dataclass
class MoveRecord:
    """一条技术记录"""
    move_id: str = ""                     # 唯一 ID（自动生成或指定）
    move_name: str = ""                   # 技术名称，如 "サンダースネーク"
    category: str = ""                    # 技术分类（技/雷蛇/ロマンス 等）
    source_video: str = ""                # 来源视频路径
    bilibili: str = ""                    # B站链接或 BV 号
    vector: Optional[np.ndarray] = None   # 特征向量
    extra: Dict = field(default_factory=dict)  # 额外信息


# ===================================================================
# 2. Faiss 中文路径兼容辅助
# ===================================================================
def _faiss_save_index(index, target_path: str):
    """绕开 Faiss 中文路径问题：先写入临时纯 ASCII 路径再移动"""
    import tempfile
    import random
    tmpdir = os.path.join(tempfile.gettempdir(), f"faiss_tmp_{random.randint(0, 999999)}")
    os.makedirs(tmpdir, exist_ok=True)
    tmp_path = os.path.join(tmpdir, "index.faiss")
    try:
        faiss.write_index(index, tmp_path)
        shutil.move(tmp_path, target_path)
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _faiss_load_index(index_path: str):
    """绕开 Faiss 中文路径问题：先复制到临时纯 ASCII 路径再读取"""
    import tempfile
    import random
    tmpdir = os.path.join(tempfile.gettempdir(), f"faiss_tmp_{random.randint(0, 999999)}")
    os.makedirs(tmpdir, exist_ok=True)
    tmp_path = os.path.join(tmpdir, "index.faiss")
    try:
        shutil.copy2(index_path, tmp_path)
        return faiss.read_index(tmp_path)
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


# ===================================================================
# 3. 向量投影器（可选：将原始特征映射到目标维度）
# ===================================================================
class FeatureProjector:
    """
    将原始特征向量投影到目标维度。
    简化版：用随机正交矩阵做近似等距映射，保留余弦相似度结构。
    正式版应替换为训练好的神经网络。
    """

    def __init__(self, input_dim: int, output_dim: int = 512, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        rng = np.random.RandomState(seed)
        # 随机正交矩阵
        A = rng.randn(output_dim, input_dim)
        Q, _ = np.linalg.qr(A)  # QR 分解取正交部分
        self.proj = Q[:, :input_dim]  # (output_dim, input_dim)

    def project(self, vec: np.ndarray) -> np.ndarray:
        """投影到目标维度"""
        vec = np.asarray(vec, dtype=np.float32).flatten()
        if len(vec) != self.input_dim:
            raise ValueError(f"输入向量维度 {len(vec)} != {self.input_dim}")
        result = self.proj @ vec
        # L2 归一化（余弦相似度）
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        return result.astype(np.float32)

    def save(self, path: str):
        np.savez(path, proj=self.proj, input_dim=self.input_dim, output_dim=self.output_dim)

    @classmethod
    def load(cls, path: str) -> "FeatureProjector":
        data = np.load(path, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.proj = data["proj"]
        obj.input_dim = int(data["input_dim"])
        obj.output_dim = int(data["output_dim"])
        return obj

    @staticmethod
    def concatenate(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """
        拼接 A路（骨骼）和 B路（光流）向量。
        这是未来的核心操作：concat → 1024维终极向量。
        """
        return np.concatenate([np.asarray(vec_a).flatten(), np.asarray(vec_b).flatten()])


# ===================================================================
# 3. Faiss 向量数据库
# ===================================================================
class WotaVectorDB:
    """
    Faiss 向量索引 + 元数据管理。

    索引策略：
      - 小数据 (<1000): IndexFlatIP（暴力内积 = 余弦相似度）
      - 大数据 (>=1000): IndexIVFFlat（倒排索引加速）
    """

    def __init__(self, dim: int, index_type: str = "auto", nlist: int = 100):
        """
        dim: 向量维度
        index_type: "auto" | "flat" | "ivf"
        nlist: IVF 聚类中心数（仅当 index_type="ivf" 时生效）
        """
        if not HAS_FAISS:
            raise RuntimeError("请先安装 faiss: pip install faiss-cpu")
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.index: Optional[faiss.Index] = None
        self._records: Dict[int, MoveRecord] = {}  # Faiss内部id → MoveRecord
        self._id_to_faiss: Dict[str, int] = {}     # move_id → Faiss内部id
        self._next_faiss_id = 0
        self._built = False

    # ---------- 索引构建 ----------
    def _create_index(self):
        """根据策略创建 Faiss 索引"""
        if self.index_type == "flat" or (self.index_type == "auto" and self._next_faiss_id < 1000):
            self.index = faiss.IndexFlatIP(self.dim)  # 内积（归一化后=余弦相似度）
            print(f"[DB] 创建 IndexFlatIP ({self.dim}维)")
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            actual_nlist = min(self.nlist, self._next_faiss_id // 5)
            actual_nlist = max(actual_nlist, 4)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)
            print(f"[DB] 创建 IndexIVFFlat ({self.dim}维, nlist={actual_nlist})")

    # ---------- 入库 ----------
    def insert(self, record: MoveRecord) -> str:
        """
        插入单条记录。自动生成 move_id（如果未提供）。
        返回 move_id。
        """
        if record.vector is None:
            raise ValueError("record.vector 不能为空")

        if not record.move_id:
            record.move_id = self._generate_id(record.move_name, record.source_video)

        vec = np.asarray(record.vector, dtype=np.float32).flatten()
        if len(vec) != self.dim:
            raise ValueError(f"向量维度 {len(vec)} != 索引维度 {self.dim}")

        # L2 归一化（内积 = 余弦相似度）
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "auto" and self._next_faiss_id >= 1000 and self.index.__class__ == faiss.IndexFlatIP:
            # 从 flat 升级到 ivf
            self._rebuild_index("ivf")

        self.index.add(vec.reshape(1, -1))
        faiss_id = self._next_faiss_id
        self._id_to_faiss[record.move_id] = faiss_id
        self._records[faiss_id] = record
        self._next_faiss_id += 1
        self._built = True

        return record.move_id

    def insert_batch(self, records: List[MoveRecord]) -> List[str]:
        """批量入库"""
        ids = []
        vecs = []
        for r in records:
            if r.vector is None:
                continue
            if not r.move_id:
                r.move_id = self._generate_id(r.move_name, r.source_video)
            vec = np.asarray(r.vector, dtype=np.float32).flatten()
            if len(vec) != self.dim:
                raise ValueError(f"向量维度 {len(vec)} != {self.dim} for {r.move_name}")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            ids.append(r.move_id)
            vecs.append(vec)

        if not vecs:
            return []

        vecs = np.array(vecs, dtype=np.float32)
        if self.index is None:
            self._create_index()

        if self.index_type in ("ivf", "auto") and isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            self.index.train(vecs)
            print("[DB] IVF 索引训练完成")

        self.index.add(vecs)

        for i, (mid, r) in enumerate(zip(ids, records)):
            faiss_id = self._next_faiss_id + i
            self._id_to_faiss[mid] = faiss_id
            self._records[faiss_id] = r

        self._next_faiss_id += len(vecs)
        self._built = True
        print(f"[DB] 批量入库 {len(vecs)} 条记录")
        return ids

    # ---------- 搜索 ----------
    def search(self, query_vec: np.ndarray, k: int = 5,
               min_score: float = 0.0) -> List[Dict]:
        """
        搜索最相似的技术。

        参数:
          query_vec: 查询向量（即 WotaOpticalFlowPipeline.get_embedding_snapshot() 的输出）
          k: 返回 Top-K
          min_score: 最低置信度阈值 (0~1)

        返回:
          [{"move_name": str, "category": str, "confidence": float, ...}, ...]
        """
        if not self._built or self.index is None:
            raise RuntimeError("数据库为空，请先 insert 数据")

        vec = np.asarray(query_vec, dtype=np.float32).flatten().reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(f"查询向量维度 {vec.shape[1]} != 索引维度 {self.dim}")

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx not in self._records:
                continue
            # IP → 余弦：score ∈ [-1, 1]，映射到 [0, 1]
            confidence = float((score + 1.0) / 2.0)
            if confidence < min_score:
                continue
            rec = self._records[idx]
            results.append({
                "move_id": rec.move_id,
                "move_name": rec.move_name,
                "category": rec.category,
                "source_video": rec.source_video,
                "bilibili": rec.bilibili,
                "confidence": round(confidence, 4),
                "raw_score": float(score),
                "extra": rec.extra,
            })

        return results

    def search_one(self, query_vec: np.ndarray, min_score: float = 0.5) -> Optional[Dict]:
        """返回最佳匹配（单个）"""
        results = self.search(query_vec, k=1, min_score=min_score)
        return results[0] if results else None

    # ---------- 管理 ----------
    def delete(self, move_id: str) -> bool:
        """根据 move_id 删除记录（注意：Faiss 不支持真正删除，仅从元数据移除）"""
        if move_id not in self._id_to_faiss:
            return False
        faiss_id = self._id_to_faiss.pop(move_id)
        self._records.pop(faiss_id, None)
        print(f"[DB] 已标记删除: {move_id}（需 rebuild 索引才能真正从向量库删除）")
        return True

    def update_move(self, move_id: str, move_name: str = None,
                    category: str = None, bilibili: str = None) -> bool:
        """修改技术详情（不改向量）"""
        if move_id not in self._id_to_faiss:
            return False
        faiss_id = self._id_to_faiss[move_id]
        rec = self._records.get(faiss_id)
        if not rec:
            return False
        if move_name is not None:
            rec.move_name = move_name
        if category is not None:
            rec.category = category
        if bilibili is not None:
            rec.bilibili = bilibili
        print(f"[DB] 已更新技术: {move_id}")
        return True

    def list_all(self) -> List[Dict]:
        """列出所有技术"""
        return [
            {"move_id": r.move_id, "move_name": r.move_name, "category": r.category, "bilibili": r.bilibili}
            for r in self._records.values()
        ]

    def count(self) -> int:
        return len(self._records)

    def _rebuild_index(self, new_type: str = "ivf"):
        """重建索引（用于更新类型或真正删除后重建）"""
        print("[DB] 正在重建索引...")
        # 先备份所有记录
        saved_records = list(self._records.values())
        vecs = []
        for rec in saved_records:
            if rec.vector is not None:
                vec = np.asarray(rec.vector, dtype=np.float32).flatten()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vecs.append(vec)

        # 重建索引
        self._records.clear()
        self._id_to_faiss.clear()
        self._next_faiss_id = 0
        self.index_type = new_type
        self.index = None
        self._create_index()

        if isinstance(self.index, faiss.IndexIVFFlat) and len(vecs) >= 4:
            self.index.train(np.array(vecs, dtype=np.float32))

        self.insert_batch(saved_records)
        print(f"[DB] 索引重建完成 ({len(self._records)} 条)")

    # ---------- 持久化 ----------
    def save(self, path: str):
        """
        保存索引和元数据到磁盘。
        生成两个文件: path.index (Faiss二进制) + path.meta (JSON)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if not self._built or self.index is None:
            raise RuntimeError("数据库为空，无法保存")

        base = path.rstrip("/").rstrip("\\")

        # Faiss 索引 — 通过临时目录绕开中文路径兼容性问题
        index_path = f"{base}.index"
        _faiss_save_index(self.index, index_path)

        # 元数据（去掉向量本身以减小文件体积）
        meta = {
            "dim": self.dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "next_faiss_id": self._next_faiss_id,
            "records": [],
            "id_to_faiss_int": {k: int(v) for k, v in self._id_to_faiss.items()},
        }
        for faiss_id, rec in self._records.items():
            meta["records"].append({
                "faiss_id": int(faiss_id),
                "move_id": rec.move_id,
                "move_name": rec.move_name,
                "category": rec.category,
                "source_video": rec.source_video,
                "bilibili": rec.bilibili,
                "extra": rec.extra,
                "vector": rec.vector.tolist() if rec.vector is not None else None,
            })
        with open(f"{base}.meta", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[DB] 已保存: {base}.index + {base}.meta ({len(self._records)} 条)")

    @classmethod
    def load(cls, path: str) -> "WotaVectorDB":
        """从磁盘加载"""
        base = path.rstrip("/").rstrip("\\")
        index_path = f"{base}.index"
        meta_path = f"{base}.meta"

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path} / {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        db = cls.__new__(cls)
        db.dim = meta["dim"]
        db.index_type = meta.get("index_type", "flat")
        db.nlist = meta.get("nlist", 100)
        db.index = _faiss_load_index(index_path)
        db._records = {}
        db._id_to_faiss = {k: int(v) for k, v in meta["id_to_faiss_int"].items()}
        db._next_faiss_id = meta["next_faiss_id"]
        db._built = True

        for r in meta["records"]:
            db._records[r["faiss_id"]] = MoveRecord(
                move_id=r["move_id"],
                move_name=r["move_name"],
                category=r.get("category", ""),
                source_video=r.get("source_video", ""),
                bilibili=r.get("bilibili", ""),
                vector=np.array(r["vector"], dtype=np.float32) if r.get("vector") else None,
                extra=r.get("extra", {}),
            )

        print(f"[DB] 已加载: {base}.index ({len(db._records)} 条)")
        return db

    @staticmethod
    def _generate_id(name: str, source: str = "") -> str:
        raw = f"{name}_{source}_{os.urandom(4).hex()}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ===================================================================
# 4. 高级检索：DTW 时序对齐 + 向量搜索（两阶段）
# ===================================================================
class WotaRetriever:
    """
    两阶段检索器（未来扩展 DTW）：
      阶段1: Faiss 粗筛 → Top-K 候选
      阶段2: DTW 精细比对（需逐帧向量序列，当前为占位）
    """

    def __init__(self, db: WotaVectorDB):
        self.db = db

    def retrieve(self, query_vec: np.ndarray, k_faiss: int = 10,
                 min_confidence: float = 0.5) -> List[Dict]:
        """阶段1：Faiss 粗筛"""
        return self.db.search(query_vec, k=k_faiss, min_score=min_confidence)

    def retrieve_with_dtw(self, query_vec: np.ndarray, query_frame_vectors: List[np.ndarray] = None,
                          k_faiss: int = 10) -> List[Dict]:
        """
        两阶段检索（未来实现）：
          阶段1: Faiss 粗筛
          阶段2: 对候选做 DTW 逐帧比对
        当前 query_frame_vectors 为 None 时，仅做阶段1。
        """
        candidates = self.db.search(query_vec, k=k_faiss)
        if not query_frame_vectors or len(candidates) <= 1:
            return candidates

        # TODO: 阶段2 DTW 精细比对
        # 需要候选视频的逐帧向量序列存储在 extra 字段中
        print("[Retriever] DTW 精细比对（待实现）")
        return candidates


# ===================================================================
# 5. 与光流模块的集成桥接
# ===================================================================
def build_vector_from_video(video_path: str, method: str = "farneback",
                            color_presets: List[str] = None,
                            step: int = 1, max_frames: int = 300,
                            target_dim: int = 512) -> np.ndarray:
    """
    便捷函数：从视频直接出向量。
    调用 optical_flow_wota.py 的 Pipeline，通过 FeatureProjector 投影到 target_dim。
    """
    from optical_flow_wota import WotaOpticalFlowPipeline

    pipeline = WotaOpticalFlowPipeline(
        video_path=video_path,
        method=method,
        color_presets=color_presets,
    )
    pipeline.run(step=step, max_frames=max_frames, output_dir="./temp_output")

    raw_vec = pipeline.get_embedding_snapshot(normalize=True)

    if target_dim > len(raw_vec):
        projector = FeatureProjector(input_dim=len(raw_vec), output_dim=target_dim)
        vec = projector.project(raw_vec)
    else:
        vec = raw_vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

    return vec


# ===================================================================
# 6. 命令行入口
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wota艺 向量数据库管理")
    sub = parser.add_subparsers(dest="cmd")

    # ---- 入库 ----
    add_parser = sub.add_parser("add", help="将标准视频入库")
    add_parser.add_argument("video", help="视频路径")
    add_parser.add_argument("-n", "--name", required=True, help="技术名称")
    add_parser.add_argument("-c", "--category", default="", help="技术分类")
    add_parser.add_argument("--bilibili", default="", help="B站视频链接或BV号")
    add_parser.add_argument("-d", "--db", default="./wota_db", help="数据库路径前缀")
    add_parser.add_argument("--dim", type=int, default=12, help="向量维度")
    add_parser.add_argument("--target-dim", type=int, default=512, help="投影目标维度")
    add_parser.add_argument("--colors", nargs="*", default=["white"], help="光棒颜色")

    # ---- 初始化数据库 ----
    init_parser = sub.add_parser("init", help="初始化空数据库（建库）")
    init_parser.add_argument("-d", "--db", default="./wota_db", help="数据库路径前缀")
    init_parser.add_argument("--dim", type=int, default=512, help="向量维度（默认512）")

    # ---- 搜索 ----
    search_parser = sub.add_parser("search", help="搜索用户视频对应的技术")
    search_parser.add_argument("video", help="用户视频路径")
    search_parser.add_argument("-d", "--db", default="./wota_db", help="数据库路径前缀")
    search_parser.add_argument("-k", type=int, default=5, help="返回Top-K")
    search_parser.add_argument("--min-score", type=float, default=0.5, help="最小置信度")
    search_parser.add_argument("--colors", nargs="*", default=["white"], help="光棒颜色")

    # ---- 列表 ----
    list_parser = sub.add_parser("list", help="列出所有技术")
    list_parser.add_argument("-d", "--db", default="./wota_db", help="数据库路径前缀")

    # ---- 删除 ----
    del_parser = sub.add_parser("delete", help="删除技术（逻辑删除）")
    del_parser.add_argument("move_id", help="技术ID")
    del_parser.add_argument("-d", "--db", default="./wota_db", help="数据库路径前缀")

    args = parser.parse_args()

    if args.cmd == "init":
        meta_path = f"{args.db}.meta"
        if os.path.exists(meta_path):
            print(f"[Init] 数据库已存在: {args.db}.meta，无需重复初始化")
            print(f"[Init] 如需重建请先删除 {args.db}.index 和 {args.db}.meta")
            exit(0)

        db = WotaVectorDB(dim=args.dim)
        db.index = faiss.IndexFlatIP(args.dim)  # 创建空索引
        db._built = True
        db.save(args.db)
        print(f"[Init] 空数据库已创建: {args.db}.index + {args.db}.meta")
        print(f"[Init] 向量维度: {args.dim}")
        print(f"[Init] 下一步: python vector_db_wota.py add <视频> -n <技术名> -c <分类>")

    elif args.cmd == "add":
        # 1. 提取向量
        print(f"[Add] 正在处理: {args.video}")
        vec = build_vector_from_video(
            video_path=args.video,
            color_presets=args.colors,
            target_dim=args.target_dim,
        )
        actual_dim = len(vec)
        print(f"[Add] 提取到 {actual_dim} 维向量")

        # 2. 打开/创建数据库
        meta_path = f"{args.db}.meta"
        if os.path.exists(meta_path):
            db = WotaVectorDB.load(args.db)
            if db.dim != actual_dim:
                print(f"[Add] 警告：向量维度 {actual_dim} != 数据库维度 {db.dim}")
        else:
            db = WotaVectorDB(dim=actual_dim)

        # 3. 入库
        record = MoveRecord(
            move_name=args.name,
            category=args.category,
            source_video=args.video,
            bilibili=args.bilibili,
            vector=vec,
        )
        mid = db.insert(record)
        db.save(args.db)
        print(f"[Add] 入库成功: {args.name} (ID: {mid})")

    elif args.cmd == "search":
        if not os.path.exists(f"{args.db}.meta"):
            print("[Search] 错误：数据库不存在，请先 add 标准视频")
            exit(1)

        db = WotaVectorDB.load(args.db)
        print(f"[Search] 数据库: {db.count()} 条技术")

        vec = build_vector_from_video(
            video_path=args.video,
            color_presets=args.colors,
            target_dim=db.dim,
        )

        results = db.search(vec, k=args.k, min_score=args.min_score)
        if not results:
            print("[Search] 未找到匹配技术（置信度低于阈值）")
        else:
            print(f"\n{'='*55}")
            print(f"  查询视频: {args.video}")
            print(f"{'='*55}")
            for i, r in enumerate(results):
                bar = "█" * int(r["confidence"] * 30)
                print(f"  {i+1}. {r['move_name']:<20s}  [{r['category']}]")
                print(f"     置信度: {r['confidence']:.2%}  {bar}")
                if r["source_video"]:
                    print(f"     标准视频: {r['source_video']}")
                print()

    elif args.cmd == "list":
        if not os.path.exists(f"{args.db}.meta"):
            print("[List] 数据库为空")
            exit(0)
        db = WotaVectorDB.load(args.db)
        print(f"\n数据库: {db.count()} 条技术")
        print("-" * 40)
        for r in db.list_all():
            print(f"  [{r['move_id']}] {r['move_name']:<20s} {r['category']}")
            if r.get("bilibili"):
                print(f"       B站: {r['bilibili']}")

    elif args.cmd == "delete":
        if not os.path.exists(f"{args.db}.meta"):
            print("[Delete] 数据库不存在")
            exit(1)
        db = WotaVectorDB.load(args.db)
        ok = db.delete(args.move_id)
        if ok:
            db.save(args.db)
            print(f"[Delete] 已删除: {args.move_id}")
        else:
            print(f"[Delete] 未找到: {args.move_id}")

    else:
        parser.print_help()
