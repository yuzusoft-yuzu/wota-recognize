"""
B站视频溯源匹配模块
====================
独立于动作识别(/recognize)的第二个模块：视频内容指纹匹配。

定位：给定用户上传的视频，在B站wota艺视频库里找出
      "完全相同/转载版本"的B站视频，返回其链接。

设计：
  - 视频指纹 = 每个B站视频抽 N 个关键帧，每帧用 FeatureBuilder 提取 38 维特征。
    一个 B站视频 = N 个 38 维向量（N 个"探针"）。
  - 索引：Faiss IndexFlatIP（内积；向量归一化后即余弦相似度）。
  - 检索：用户视频同样抽关键帧提特征 → 每帧在 Faiss 找最近邻 →
    按 bvid 聚合投票（命中次数 + 最高相似度）→ 返回 Top-K 的 B站链接。
  - 特征维度：与 skeleton_light_fusion.py 的 FeatureBuilder.DIM 一致（38）。

数据文件（data/ 目录，不入 git）：
  - bilibili_index.faiss       : Faiss 索引（向量）
  - bilibili_vector_meta.db    : SQLite，vector_id → bvid/title/link/帧序号
  - bilibili_videos.db         : B站视频元数据（爬虫产出）
"""

from __future__ import annotations

import os
import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# ---------- 路径 ----------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = str(DATA_DIR / "bilibili_index.faiss")
META_DB_PATH = str(DATA_DIR / "bilibili_vector_meta.db")
VIDEOS_DB_PATH = str(DATA_DIR / "bilibili_videos.db")

# 特征维度（与 skeleton_light_fusion.FeatureBuilder.DIM 一致）
_FEATURE_DIM = 38

# Faiss 可选：未安装时优雅降级（仅无法检索，接口仍可响应"库未就绪"）
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


class BilibiliMatcher:
    """B站视频溯源匹配器：加载 Faiss 索引 + 元数据，对外提供 match() 检索。

    线程安全：索引加载后只读，检索可并发。构建索引用独立方法 build()。
    """

    def __init__(self, index_path: str = INDEX_PATH,
                 meta_db_path: str = META_DB_PATH):
        self.index_path = index_path
        self.meta_db_path = meta_db_path
        self._index = None          # faiss.Index
        self._meta: Dict[int, Dict[str, Any]] = {}  # vector_id -> {bvid,title,link,...}
        self._lock = threading.Lock()
        self._loaded = False

    # ---------- 加载 ----------
    def load(self) -> bool:
        """加载索引与元数据。返回是否就绪（库为空/文件缺失返回 False）。"""
        with self._lock:
            if self._loaded:
                return self._index is not None
            if not _HAS_FAISS:
                return False
            if not (os.path.exists(self.index_path) and os.path.exists(self.meta_db_path)):
                return False
            try:
                self._index = faiss.read_index(self.index_path)
                self._meta = self._load_meta()
                self._loaded = True
                return self._index is not None and len(self._meta) > 0
            except Exception:
                self._index = None
                self._meta = {}
                return False

    def _load_meta(self) -> Dict[int, Dict[str, Any]]:
        if not os.path.exists(self.meta_db_path):
            return {}
        conn = sqlite3.connect(self.meta_db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT vector_id, bvid, title, up_name, play_count, link, frame_idx "
                "FROM vector_meta"
            ).fetchall()
            return {r["vector_id"]: dict(r) for r in rows}
        finally:
            conn.close()

    # ---------- 状态 ----------
    @property
    def ready(self) -> bool:
        if not self._loaded:
            return self.load()
        return self._index is not None and len(self._meta) > 0

    @property
    def vector_count(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    @property
    def video_count(self) -> int:
        return len({m.get("bvid") for m in self._meta.values()})

    # ---------- 检索 ----------
    def match(self, query_vectors: List[List[float]],
              top_k: int = 5, per_frame_k: int = 5,
              min_sim: float = 0.0) -> List[Dict[str, Any]]:
        """对用户视频的关键帧特征做溯源匹配，返回 Top-K 的 B站视频。

        query_vectors: 用户视频的关键帧特征列表（每个 38 维）
        top_k: 返回前 K 个B站视频
        per_frame_k: 每帧在 Faiss 中取多少近邻参与投票
        min_sim: 低于此相似度的候选过滤掉
        返回: [{"bvid","title","link","up_name","play_count",
                "similarity","vote","vote_fraction"}, ...] 按 similarity 降序
        """
        if not self.ready or not query_vectors:
            return []
        q = np.asarray(query_vectors, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != _FEATURE_DIM:
            return []
        # 归一化（用内积近似余弦相似度）
        faiss.normalize_L2(q)
        k = min(per_frame_k, self._index.ntotal)
        if k <= 0:
            return []
        # 每帧检索 k 个近邻
        scores, ids = self._index.search(q, k)  # scores: [n_frames, k]
        # 按 bvid 聚合投票
        bvid_stat: Dict[str, Dict[str, Any]] = {}
        total_votes = 0
        for fi in range(scores.shape[0]):
            for ni in range(scores.shape[1]):
                vid = int(ids[fi, ni])
                if vid < 0 or vid not in self._meta:
                    continue
                sim = float(scores[fi, ni])
                if sim < min_sim:
                    continue
                meta = self._meta[vid]
                bvid = meta["bvid"]
                st = bvid_stat.setdefault(bvid, {
                    "bvid": bvid, "title": meta.get("title", ""),
                    "link": meta.get("link", ""), "up_name": meta.get("up_name", ""),
                    "play_count": meta.get("play_count", 0),
                    "best_sim": sim, "vote": 0,
                })
                st["vote"] += 1
                total_votes += 1
                if sim > st["best_sim"]:
                    st["best_sim"] = sim
        if not bvid_stat:
            return []
        # 综合排序：相似度为主(0.7) + 投票占比(0.3)
        n_frames = q.shape[0]
        results = []
        for st in bvid_stat.values():
            vote_frac = st["vote"] / max(n_frames, 1)
            combined = 0.7 * st["best_sim"] + 0.3 * vote_frac
            results.append({
                "bvid": st["bvid"], "title": st["title"], "link": st["link"],
                "up_name": st["up_name"], "play_count": st["play_count"],
                "similarity": round(st["best_sim"] * 100, 1),
                "vote": st["vote"],
                "vote_fraction": round(vote_frac * 100, 1),
                "match": round(combined * 100, 1),
            })
        results.sort(key=lambda r: r["match"], reverse=True)
        return results[:top_k]


# ---------- 单例 ----------
_matcher_instance: Optional[BilibiliMatcher] = None
_matcher_lock = threading.Lock()


def get_matcher() -> BilibiliMatcher:
    """全局单例 BilibiliMatcher（懒加载）。"""
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = BilibiliMatcher()
    return _matcher_instance


# ---------- 关键帧特征提取（供 app.py 的 /match-video 调用） ----------
def extract_keyframe_features(video_path: str, n_frames: int = 5,
                              use_skeleton: bool = True) -> List[List[float]]:
    """从用户视频均匀抽 n_frames 个关键帧，每帧提取 38 维特征。

    复用 skeleton_light_fusion 的逐帧提取链路（增强→骨骼→光斑→特征），
    保证与B站视频特征库同源可比。
    返回: [[38维], ...] 长度 <= n_frames
    """
    from skeleton_light_fusion import (
        FrameSampler, CLAHEEnhancer, SkeletonExtractor, LightDetector,
        FeatureBuilder, Skeleton,
    )
    import cv2

    sampler = FrameSampler(step=1, max_frames=n_frames)
    frames, _ = sampler.sample(video_path)
    if not frames:
        return []
    # 均匀采样到 n_frames
    if len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        frames = [frames[i] for i in idx]

    enhancer = CLAHEEnhancer()
    detector = LightDetector()
    builder = FeatureBuilder()
    skeleton = SkeletonExtractor() if use_skeleton else None

    feats: List[List[float]] = []
    prev_l = prev_r = None
    try:
        for frame in frames:
            enhanced = enhancer.enhance(frame)
            sk = skeleton.extract(enhanced) if skeleton is not None else Skeleton()
            l_light = detector.detect_near_wrist(enhanced, sk.l_wrist, sk.l_wrist_vis)
            r_light = detector.detect_near_wrist(enhanced, sk.r_wrist, sk.r_wrist_vis)
            vec = builder.build(sk, l_light, r_light, prev_l, prev_r)
            feats.append(vec)
            prev_l, prev_r = l_light, r_light
    finally:
        if skeleton is not None:
            skeleton.close()
    return feats
