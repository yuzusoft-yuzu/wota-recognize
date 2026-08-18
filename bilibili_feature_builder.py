"""
B站视频关键帧下载 + 特征提取 + Faiss 索引构建
================================================
合规声明：
  - 仅下载视频用于本地抽关键帧提取特征，提取后立即删除视频文件
  - 不保留、不分发下载的视频
  - 控制频率，每次下载随机 sleep 3-8 秒
  - 用途：本地特征提取和算法测试，不商用

流程：
  1. 遍历 data/bilibili_videos.db 中每个B站视频
  2. 用 yt-dlp 下载视频到临时文件（低画质即可，省带宽）
  3. OpenCV 均匀抽 N 个关键帧
  4. 每帧用 skeleton_light_fusion.FeatureBuilder 提取 38 维特征
  5. 删除临时视频文件
  6. 全部处理完后构建 Faiss 索引 + 向量元数据

产出：
  data/bilibili_index.faiss       : Faiss IndexFlatIP（向量）
  data/bilibili_vector_meta.db    : vector_id -> bvid/title/link/帧序号

依赖：yt-dlp（pip install yt-dlp），需能访问B站

用法：
  python bilibili_feature_builder.py                  # 处理所有未处理视频
  python bilibili_feature_builder.py --limit 50       # 只处理50个
  python bilibili_feature_builder.py --build-only     # 只重建Faiss索引不下载
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
import sqlite3
import subprocess
import tempfile
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# ---------- 路径 ----------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
VIDEOS_DB_PATH = str(DATA_DIR / "bilibili_videos.db")
INDEX_PATH = str(DATA_DIR / "bilibili_index.faiss")
META_DB_PATH = str(DATA_DIR / "bilibili_vector_meta.db")
PROGRESS_PATH = str(DATA_DIR / "feature_progress.json")

FEATURE_DIM = 38
N_KEYFRAMES = 5       # 每视频抽 5 个关键帧
SLEEP_MIN = 3.0       # 下载间隔（秒）
SLEEP_MAX = 8.0

# Faiss
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


# ---------- 进度 ----------
def load_progress() -> Dict[str, Any]:
    if os.path.exists(PROGRESS_PATH):
        try:
            with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_bvids": []}


def save_progress(p: Dict[str, Any]):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


# ---------- 视频列表 ----------
def get_videos_to_process(limit: int, processed: set) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(VIDEOS_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT bvid, title, author, play_count, cover, link, tier_name "
        "FROM videos ORDER BY play_count DESC"
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        if r["bvid"] in processed:
            continue
        out.append(dict(r))
        if len(out) >= limit:
            break
    return out


# ---------- 关键帧特征提取 ----------
def download_and_extract(bvid: str, link: str, n_frames: int = N_KEYFRAMES
                         ) -> Optional[List[List[float]]]:
    """用 yt-dlp 下载视频到临时文件，OpenCV 抽关键帧，提取38维特征。
    下载的视频处理完立即删除。失败返回 None。"""
    import cv2

    tmp_dir = tempfile.mkdtemp(prefix="bilibili_")
    tmp_video = os.path.join(tmp_dir, f"{bvid}.mp4")
    try:
        # yt-dlp 下载（低画质，省带宽；B站需要 referer）
        cmd = [
            "yt-dlp", "-f", "best[height<=480]/best",
            "-o", tmp_video,
            "--no-warnings", "--no-playlist", "--no-progress",
            "--no-check-certificate",
            "--add-header", "Referer:https://www.bilibili.com/",
            link,
        ]
        ret = subprocess.run(cmd, capture_output=True, timeout=180, text=True)
        if ret.returncode != 0 or not os.path.exists(tmp_video):
            stderr = (ret.stderr or "")[-200:]
            print(f"    [yt-dlp失败] {bvid}: {stderr.strip()[:80]}")
            return None

        # OpenCV 抽关键帧
        cap = cv2.VideoCapture(tmp_video)
        if not cap.isOpened():
            print(f"    [打开失败] {bvid}")
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < n_frames:
            # 帧数不足，取所有帧
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
        else:
            idxs = np.linspace(0, total - 1, n_frames).astype(int)
            frames = []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = cap.read()
                if ok:
                    frames.append(fr)
            cap.release()

        if not frames:
            print(f"    [无帧] {bvid}")
            return None

        # 提取特征（复用 skeleton_light_fusion 链路）
        from skeleton_light_fusion import (
            CLAHEEnhancer, SkeletonExtractor, LightDetector,
            FeatureBuilder, Skeleton,
        )
        enhancer = CLAHEEnhancer()
        detector = LightDetector()
        builder = FeatureBuilder()
        # 视频溯源用关键帧，不需要骨骼也能比（骨骼是辅助）。
        # 但为与用户上传同源，开启骨骼（失败自动降级）。
        skeleton = SkeletonExtractor()

        feats: List[List[float]] = []
        prev_l = prev_r = None
        try:
            for frame in frames:
                enhanced = enhancer.enhance(frame)
                sk = skeleton.extract(enhanced)
                l_light = detector.detect_near_wrist(enhanced, sk.l_wrist, sk.l_wrist_vis)
                r_light = detector.detect_near_wrist(enhanced, sk.r_wrist, sk.r_wrist_vis)
                vec = builder.build(sk, l_light, r_light, prev_l, prev_r)
                feats.append(vec)
                prev_l, prev_r = l_light, r_light
        finally:
            skeleton.close()
        return feats

    except subprocess.TimeoutExpired:
        print(f"    [超时] {bvid}")
        return None
    except Exception as e:
        print(f"    [异常] {bvid}: {type(e).__name__}: {e}")
        return None
    finally:
        # 删除临时视频文件（合规：不保留）
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------- Faiss 索引构建 ----------
def build_index(all_vectors: List[np.ndarray], all_meta: List[Dict[str, Any]]):
    """构建 Faiss IndexFlatIP + 元数据db。"""
    if not _HAS_FAISS:
        print("[错误] 未安装 faiss，无法构建索引")
        return
    if not all_vectors:
        print("[警告] 无向量，跳过索引构建")
        return

    vecs = np.asarray(all_vectors, dtype=np.float32)
    faiss.normalize_L2(vecs)  # 归一化（内积=余弦相似度）

    index = faiss.IndexFlatIP(FEATURE_DIM)
    index.add(vecs)
    faiss.write_index(index, INDEX_PATH)
    print(f"Faiss 索引已写入: {INDEX_PATH} ({index.ntotal} 个向量)")

    # 元数据 db
    conn = sqlite3.connect(META_DB_PATH)
    conn.execute("DROP TABLE IF EXISTS vector_meta")
    conn.execute("""
        CREATE TABLE vector_meta (
            vector_id INTEGER PRIMARY KEY,
            bvid TEXT, title TEXT, up_name TEXT,
            play_count INTEGER, link TEXT, frame_idx INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO vector_meta (vector_id,bvid,title,up_name,play_count,link,frame_idx) "
        "VALUES (?,?,?,?,?,?,?)",
        [(i, m["bvid"], m.get("title", ""), m.get("author", ""),
          int(m.get("play_count", 0)), m.get("link", ""), m.get("frame_idx", 0))
         for i, m in enumerate(all_meta)],
    )
    conn.commit()
    conn.close()
    print(f"元数据已写入: {META_DB_PATH} ({len(all_meta)} 条)")


def rebuild_index_only():
    """从已处理视频的特征重建Faiss索引（不重新下载）。
    需要特征缓存文件 data/feature_cache.db。"""
    cache_db = str(DATA_DIR / "feature_cache.db")
    if not os.path.exists(cache_db):
        print("[错误] 无特征缓存，需先运行完整提取")
        return
    conn = sqlite3.connect(cache_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT bvid, title, author, play_count, link, frame_idx, feature_json "
        "FROM feature_cache ORDER BY id"
    ).fetchall()
    conn.close()

    all_vecs, all_meta = [], []
    for r in rows:
        feat = json.loads(r["feature_json"])
        all_vecs.append(np.asarray(feat, dtype=np.float32))
        all_meta.append(dict(r))
    build_index(all_vecs, all_meta)


# ---------- 特征缓存（断点续传）----------
def init_feature_cache():
    cache_db = str(DATA_DIR / "feature_cache.db")
    conn = sqlite3.connect(cache_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bvid TEXT, title TEXT, author TEXT,
            play_count INTEGER, link TEXT, frame_idx INTEGER,
            feature_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bvid ON feature_cache(bvid)")
    conn.commit()
    conn.close()


def cache_features(bvid: str, meta: Dict[str, Any], feats: List[List[float]]):
    cache_db = str(DATA_DIR / "feature_cache.db")
    conn = sqlite3.connect(cache_db)
    for fi, f in enumerate(feats):
        conn.execute(
            "INSERT INTO feature_cache (bvid,title,author,play_count,link,frame_idx,feature_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (bvid, meta.get("title", ""), meta.get("author", ""),
             int(meta.get("play_count", 0)), meta.get("link", ""), fi,
             json.dumps(f)),
        )
    conn.commit()
    conn.close()


def load_cached_vectors() -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    cache_db = str(DATA_DIR / "feature_cache.db")
    if not os.path.exists(cache_db):
        return [], []
    conn = sqlite3.connect(cache_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT bvid,title,author,play_count,link,frame_idx,feature_json "
        "FROM feature_cache ORDER BY id"
    ).fetchall()
    conn.close()
    vecs, metas = [], []
    for r in rows:
        vecs.append(np.asarray(json.loads(r["feature_json"]), dtype=np.float32))
        metas.append(dict(r))
    return vecs, metas


# ---------- 主流程 ----------
def run(limit: int = 10**9):
    init_feature_cache()
    progress = load_progress()
    processed = set(progress["processed_bvids"])
    print(f"已处理视频: {len(processed)} 个")

    videos = get_videos_to_process(limit, processed)
    print(f"待处理视频: {len(videos)} 个")

    success = 0
    for i, v in enumerate(videos):
        bvid = v["bvid"]
        print(f"\n[{i+1}/{len(videos)}] {bvid} (播放 {v['play_count']}) {v['title'][:25]}")
        feats = download_and_extract(bvid, v["link"])
        if feats:
            cache_features(bvid, v, feats)
            success += 1
            print(f"    提取 {len(feats)} 帧 OK")
        else:
            print(f"    跳过（提取失败）")
        processed.add(bvid)
        progress["processed_bvids"] = list(processed)
        save_progress(progress)
        if i < len(videos) - 1:
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    print(f"\n提取完成: 成功 {success}/{len(videos)}")

    # 构建索引
    print("\n构建 Faiss 索引...")
    vecs, metas = load_cached_vectors()
    build_index(vecs, metas)
    print(f"索引向量总数: {len(vecs)}")


def main():
    parser = argparse.ArgumentParser(description="B站视频关键帧特征提取 + Faiss索引构建")
    parser.add_argument("--limit", type=int, default=10**9, help="最多处理多少个视频")
    parser.add_argument("--build-only", action="store_true", help="只重建索引不下载")
    args = parser.parse_args()

    if args.build_only:
        rebuild_index_only()
    else:
        run(limit=args.limit)


if __name__ == "__main__":
    main()
