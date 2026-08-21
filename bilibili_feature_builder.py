"""
B站视频关键帧下载 + dHash 感知哈希提取 + 溯源库构建
====================================================
合规声明：
  - 仅下载视频用于本地抽关键帧提取 dHash，提取后立即删除视频文件
  - 不保留、不分发下载的视频
  - 控制频率，每次下载随机 sleep 3-8 秒
  - 用途：本地特征提取和算法测试，不商用

流程：
  1. 遍历 data/bilibili_videos.db 中每个B站视频
  2. 用 yt-dlp 下载视频流到临时文件（低画质）
  3. OpenCV 均匀抽 N 个关键帧（跳过过暗帧）
  4. 每帧提取 dHash 感知哈希（256位）
  5. 删除临时视频文件
  6. 全部处理完后写入 data/bilibili_dhash.db

产出：
  data/bilibili_dhash.db : 每帧 dHash + 元数据（bvid/title/link/up/play）

依赖：yt-dlp（pip install yt-dlp），需能访问B站

用法：
  python bilibili_feature_builder.py                  # 处理所有未处理视频
  python bilibili_feature_builder.py --limit 50       # 只处理50个
"""

from __future__ import annotations

import os
import time
import json
import random
import sqlite3
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import cv2


def safe_print(msg):
    """安全打印：控制台是 GBK 时对特殊字符(如⚡emoji)会崩溃，用 replace 兜底。"""
    try:
        print(msg)
    except UnicodeEncodeError:
        s = str(msg)
        print(s.encode("gbk", errors="replace").decode("gbk", errors="replace"))


# ---------- 路径 ----------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
VIDEOS_DB_PATH = str(DATA_DIR / "bilibili_videos.db")
DHASH_DB_PATH = str(DATA_DIR / "bilibili_dhash.db")
PROGRESS_PATH = str(DATA_DIR / "feature_progress.json")

# dHash 参数
HASH_SIZE = 16              # 16x16 -> 256 位
N_KEYFRAMES = 60            # 每视频抽 60 帧（密度足够支撑 1/3 片段溯源）
MIN_BRIGHTNESS = 10.0       # 跳过平均亮度 < 10 的全黑帧
SLEEP_MIN = 3.0
SLEEP_MAX = 8.0


# ---------- dHash ----------
def dhash(frame_bgr: np.ndarray) -> int:
    """计算一帧的 dHash（256位整数）。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE))
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | (1 if b else 0)
    return val


def _brightness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    """计算一帧的 HSV 量化颜色直方图（512维，归一化）。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8), [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# ---------- 数据库 ----------
def init_dhash_db():
    conn = sqlite3.connect(DHASH_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dhash (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bvid TEXT, title TEXT, up_name TEXT,
            play_count INTEGER, link TEXT,
            frame_idx INTEGER, hash_hex TEXT,
            hist_hex TEXT
        )
    """)
    # 兼容旧库：若表已存在但无 hist_hex 列，则补上
    cols = [r[1] for r in conn.execute("PRAGMA table_info(dhash)").fetchall()]
    if "hist_hex" not in cols:
        conn.execute("ALTER TABLE dhash ADD COLUMN hist_hex TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bvid ON dhash(bvid)")
    conn.commit()
    conn.close()


def save_dhash(bvid: str, meta: Dict[str, Any], feats: List[Tuple[int, np.ndarray]]):
    """保存每帧的 (dHash, 颜色直方图)。feats: [(dhash_int, hist_array), ...]"""
    conn = sqlite3.connect(DHASH_DB_PATH)
    for fi, (h, hist) in enumerate(feats):
        hist_hex = hist.tobytes().hex() if hist is not None else None
        conn.execute(
            "INSERT INTO dhash (bvid,title,up_name,play_count,link,frame_idx,hash_hex,hist_hex) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (bvid, meta.get("title", ""), meta.get("author", ""),
             int(meta.get("play_count", 0)), meta.get("link", ""),
             fi, format(h, "064x"), hist_hex),
        )
    conn.commit()
    conn.close()


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


# ---------- 下载 + 抽帧 + dHash ----------
def download_and_extract(bvid: str, link: str, n_frames: int = N_KEYFRAMES
                         ) -> Optional[List[int]]:
    """下载视频流，抽关键帧，提取 dHash，删除视频。返回 dHash int 列表。"""
    tmp_root = DATA_DIR / ".tmp_frames"
    tmp_root.mkdir(parents=True, exist_ok=True)
    ytdlp_cache = str(tmp_root / ".ytdlp_cache")
    os.makedirs(ytdlp_cache, exist_ok=True)
    tmp_video = str(tmp_root / f"{bvid}.mp4")
    for p in (tmp_video, tmp_video + ".part"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    try:
        cmd = [
            "yt-dlp", "-f", "bestvideo[height<=640]/bestvideo[height<=480]/bestvideo/best",
            "-o", tmp_video,
            "--no-warnings", "--no-playlist", "--no-progress",
            "--no-check-certificate",
            "--cache-dir", ytdlp_cache,
            "--socket-timeout", "20",
            "--retries", "1",
            "--fragment-retries", "1",
            "--add-header", "Referer:https://www.bilibili.com/",
            link,
        ]
        ret = subprocess.run(cmd, capture_output=True, timeout=120,
                             text=True, encoding="utf-8", errors="replace")
        if ret.returncode != 0 or not os.path.exists(tmp_video):
            stderr = (ret.stderr or "")[-200:]
            safe_print(f"    [yt-dlp失败] {bvid}: {stderr.strip()[:80]}")
            return None

        cap = cv2.VideoCapture(tmp_video)
        if not cap.isOpened():
            safe_print(f"    [打开失败] {bvid}")
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total <= 0:
            return None

        # 抽 2*n_frames 个候选帧（跳过暗帧后有冗余）
        cap = cv2.VideoCapture(tmp_video)
        idxs = np.linspace(0, total - 1, min(n_frames * 2, total)).astype(int)
        feats: List[Tuple[int, np.ndarray]] = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                continue
            if _brightness(fr) < MIN_BRIGHTNESS:
                continue
            feats.append((dhash(fr), hsv_hist(fr)))
            if len(feats) >= n_frames:
                break
        cap.release()
        return feats if feats else None

    except subprocess.TimeoutExpired:
        safe_print(f"    [超时] {bvid}")
        return None
    except Exception as e:
        safe_print(f"    [异常] {bvid}: {type(e).__name__}: {e}")
        return None
    finally:
        try:
            if os.path.exists(tmp_video):
                os.remove(tmp_video)
            if os.path.exists(tmp_video + ".part"):
                os.remove(tmp_video + ".part")
        except OSError:
            pass


# ---------- 主流程 ----------
def run(limit: int = 10**9):
    init_dhash_db()
    progress = load_progress()
    processed = set(progress["processed_bvids"])
    safe_print(f"已处理视频: {len(processed)} 个")

    videos = get_videos_to_process(limit, processed)
    safe_print(f"待处理视频: {len(videos)} 个")

    success = 0
    for i, v in enumerate(videos):
        bvid = v["bvid"]
        safe_print(f"\n[{i+1}/{len(videos)}] {bvid} (播放 {v['play_count']}) {v['title'][:25]}")
        hashes = download_and_extract(bvid, v["link"])
        if hashes:
            save_dhash(bvid, v, hashes)
            success += 1
            safe_print(f"    提取 {len(hashes)} 帧 dHash OK")
        else:
            safe_print(f"    跳过（提取失败）")
        processed.add(bvid)
        progress["processed_bvids"] = list(processed)
        save_progress(progress)
        if i < len(videos) - 1:
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    safe_print(f"\n提取完成: 成功 {success}/{len(videos)}")
    # 统计
    conn = sqlite3.connect(DHASH_DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM dhash").fetchone()[0]
    bvids = conn.execute("SELECT COUNT(DISTINCT bvid) FROM dhash").fetchone()[0]
    conn.close()
    safe_print(f"dHash库: {n} 帧, {bvids} 个视频")


def main():
    parser = argparse.ArgumentParser(description="B站视频 dHash 溯源库构建")
    parser.add_argument("--limit", type=int, default=10**9, help="最多处理多少个视频")
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
