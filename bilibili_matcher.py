"""
B站视频溯源匹配模块（dHash 感知哈希版）
========================================
定位：给定用户上传的视频，在B站wota艺视频库里找出
      "完全相同/转载版本/片段"的B站视频，返回其链接。

原理（经实验验证）：
  - 每个视频抽 N 个关键帧，每帧提取 dHash 感知哈希（256位）。
  - dHash 对同一视频的不同片段（如 1/3 片段）汉明距离极小（3~18），
    对不同视频汉明距离很大（>100），可精准区分"同一视频"vs"不同视频"。
  - 匹配：用户视频关键帧的 dHash 与库中所有哈希做汉明距离比对，
    距离 < 阈值的帧视为"命中"，按 bvid 聚合投票，
    命中帧数 >= min_votes 判定为同一视频，返回 Top-K 的B站链接。

数据文件（data/ 目录，不入 git）：
  - bilibili_dhash.db : 每帧 dHash + 元数据（bvid/title/link/up/play）
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2

# ---------- 路径 ----------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DHASH_DB_PATH = str(DATA_DIR / "bilibili_dhash.db")

# dHash 参数
HASH_SIZE = 16          # 16x16 -> 256 位
HASH_BITS = HASH_SIZE * HASH_SIZE

# 匹配参数（经实验确定：同视频片段距离 3~18，不同视频 >100）
DEFAULT_THRESHOLD = 40    # 汉明距离 < 40 视为"同一帧"
DEFAULT_MIN_VOTES = 3     # 至少 3 帧命中才判定为同一视频
DEFAULT_N_FRAMES = 30     # 查询视频抽 30 帧（配合库60帧/视频的密度）


# ---------- dHash 感知哈希 ----------
def dhash(frame_bgr: np.ndarray) -> int:
    """计算一帧的 dHash（256位整数）。
    灰度 → 缩放到 (16+1)x16 → 相邻列比较 → 256位位串。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE))
    diff = resized[:, 1:] > resized[:, :-1]  # bool 数组 (16,16)
    bits = diff.flatten()
    # 打包成 int（256位）
    val = 0
    for b in bits:
        val = (val << 1) | (1 if b else 0)
    return val


def hamming(a: int, b: int) -> int:
    """两个 256 位整数的汉明距离（不同位数）。"""
    return (a ^ b).bit_count()


def _brightness(frame_bgr: np.ndarray) -> float:
    """帧平均亮度 0~255，用于跳过全黑帧。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def extract_query_hashes(video_path: str, n_frames: int = DEFAULT_N_FRAMES,
                         min_brightness: float = 10.0) -> List[int]:
    """从用户视频均匀抽 n_frames 个关键帧，每帧提取 dHash。
    跳过过暗的帧（全黑帧 dHash 会误匹配）。返回 256位 int 列表。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(n_frames * 2, total)).astype(int)
    hashes: List[int] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        if _brightness(fr) < min_brightness:
            continue  # 跳过过暗帧
        hashes.append(dhash(fr))
        if len(hashes) >= n_frames:
            break
    cap.release()
    return hashes


# ---------- 匹配器 ----------
class BilibiliMatcher:
    """dHash 视频溯源匹配器：加载库哈希，汉明距离投票匹配。"""

    def __init__(self, db_path: str = DHASH_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._loaded = False
        self._mtime = 0.0  # 数据库文件修改时间，用于热更新检测
        self._hashes: List[Tuple[int, Dict[str, Any]]] = []  # [(hash_int, meta), ...]
        self._bvid_index: Dict[str, Dict[str, Any]] = {}     # bvid -> 视频元数据

    def _db_mtime(self) -> float:
        try:
            return os.path.getmtime(self.db_path)
        except OSError:
            return 0.0

    def load(self) -> bool:
        """加载 dHash 库。返回是否就绪。"""
        with self._lock:
            if self._loaded:
                return len(self._hashes) > 0
            return self._do_load()

    def reload(self) -> bool:
        """强制重新加载库（热更新用）。返回是否就绪。"""
        with self._lock:
            return self._do_load()

    def _do_load(self) -> bool:
        if not os.path.exists(self.db_path):
            self._loaded = False
            self._hashes = []
            self._bvid_index = {}
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT bvid, title, up_name, play_count, link, frame_idx, hash_hex "
                "FROM dhash ORDER BY bvid, frame_idx"
            ).fetchall()
            conn.close()
            self._hashes = []
            self._bvid_index = {}
            for r in rows:
                meta = {
                    "bvid": r["bvid"], "title": r["title"] or "",
                    "up_name": r["up_name"] or "", "play_count": int(r["play_count"] or 0),
                    "link": r["link"] or "",
                }
                self._hashes.append((int(r["hash_hex"], 16), meta))
                if r["bvid"] not in self._bvid_index:
                    self._bvid_index[r["bvid"]] = meta
            self._loaded = True
            self._mtime = self._db_mtime()
            return len(self._hashes) > 0
        except Exception:
            self._loaded = False
            self._hashes = []
            self._bvid_index = {}
            return False

    def refresh_if_changed(self) -> bool:
        """检测数据库文件是否变化，变化则热重载。返回库是否就绪。"""
        current = self._db_mtime()
        if current > 0 and current != self._mtime:
            return self.reload()
        return self.ready

    @property
    def ready(self) -> bool:
        if not self._loaded:
            return self.load()
        return len(self._hashes) > 0

    @property
    def video_count(self) -> int:
        return len(self._bvid_index)

    @property
    def hash_count(self) -> int:
        return len(self._hashes)

    def match(self, query_hashes: List[int],
              top_k: int = 5, threshold: int = DEFAULT_THRESHOLD,
              min_votes: int = DEFAULT_MIN_VOTES) -> List[Dict[str, Any]]:
        """对用户视频的 dHash 序列做溯源匹配，返回 Top-K 的B站视频。

        query_hashes: 用户视频关键帧的 dHash 列表（256位 int）
        threshold: 汉明距离 < threshold 视为同一帧
        min_votes: 至少 min_votes 帧命中才判定同一视频
        返回: [{"bvid","title","link","up_name","play_count",
                "votes","min_distance","similarity"}, ...]
        """
        if not self.ready or not query_hashes:
            return []
        # 逐帧比对所有库哈希，找命中
        bvid_hits: Dict[str, Dict[str, Any]] = {}
        for qh in query_hashes:
            for hh, meta in self._hashes:
                d = hamming(qh, hh)
                if d < threshold:
                    bvid = meta["bvid"]
                    st = bvid_hits.get(bvid)
                    if st is None:
                        st = {
                            "bvid": bvid, "title": meta["title"],
                            "link": meta["link"], "up_name": meta["up_name"],
                            "play_count": meta["play_count"],
                            "votes": 0, "min_distance": d,
                        }
                        bvid_hits[bvid] = st
                    st["votes"] += 1
                    if d < st["min_distance"]:
                        st["min_distance"] = d
        if not bvid_hits:
            return []
        results = []
        for st in bvid_hits.values():
            if st["votes"] < min_votes:
                continue
            # 相似度 = 100 * (1 - min_distance/256)，直观百分比
            similarity = round(100.0 * (1.0 - st["min_distance"] / HASH_BITS), 1)
            results.append({
                "bvid": st["bvid"], "title": st["title"], "link": st["link"],
                "up_name": st["up_name"], "play_count": st["play_count"],
                "votes": st["votes"], "min_distance": st["min_distance"],
                "similarity": similarity,
            })
        # 排序：命中帧数降序，其次最小距离升序
        results.sort(key=lambda r: (-r["votes"], r["min_distance"]))
        return results[:top_k]


# ---------- 单例 ----------
_matcher_instance: Optional[BilibiliMatcher] = None
_matcher_lock = threading.Lock()


def get_matcher() -> BilibiliMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = BilibiliMatcher()
    return _matcher_instance
