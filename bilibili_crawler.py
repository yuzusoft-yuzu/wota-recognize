"""
B站 Wota艺 视频元数据爬虫
==========================
合规声明：
  - 仅爬取B站公开的元数据（标题/链接/播放量/UP主/封面/BVID）
  - 不下载完整视频（关键帧下载在 bilibili_feature_builder.py，且提取特征后即删）
  - 控制频率：每次请求随机 sleep 2-5 秒，单线程
  - 用途：本地特征提取和算法测试，不商用、不分发
  - 遵守B站 ToS，如遇反爬限制则暂停

按播放量分梯队爬取：>1万 → >5千 → >1千 → >700 → >500 → >300 → <300
遇到重复视频(BVID)跳过。支持断点续爬。

数据存储：data/bilibili_videos.db (SQLite)
  表 videos: bvid(PK), title, author, play_count, cover, link, tier, crawled_at

用法：
  python bilibili_crawler.py                  # 从头/断点继续爬
  python bilibili_crawler.py --max-per-tier 200  # 每个梯队最多爬200个
  python bilibili_crawler.py --tier 10000     # 只爬某个梯队
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
import sqlite3
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote

import urllib.request
import urllib.error

# ---------- 配置 ----------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DB_PATH = str(DATA_DIR / "bilibili_videos.db")
STATE_PATH = str(DATA_DIR / "crawler_state.json")

DATA_DIR.mkdir(parents=True, exist_ok=True)

# 播放量梯队（从高到低）：(下限, 上限, 梯队名)
TIERS: List[Tuple[int, int, str]] = [
    (10000, 10**9, "1万以上"),
    (5000, 10000, "5千-1万"),
    (1000, 5000, "1千-5千"),
    (700, 1000, "700-1千"),
    (500, 700, "500-700"),
    (300, 500, "300-500"),
    (0, 300, "300以下"),
]

# 搜索关键词（覆盖 wota艺 的不同叫法）
KEYWORDS = [
    "wota艺", "ヲタ芸", "打call", "荧光棒舞", "应援打call",
    "wotagei", "otasigei", "御宅艺", "宅艺",
]

# B站搜索 API
SEARCH_URL = "https://api.bilibili.com/x/web-interface/search/type"
# 模拟浏览器 UA（避免被简单拦截）
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# 频率控制
SLEEP_MIN = 2.0
SLEEP_MAX = 5.0
MAX_PAGES_PER_KEYWORD = 50       # 每个关键词最多翻 50 页
PAGE_SIZE = 20                    # B站搜索每页约 20 条


# ---------- 数据库 ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            bvid TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            play_count INTEGER,
            cover TEXT,
            link TEXT,
            tier_name TEXT,
            crawled_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_play ON videos(play_count)")
    conn.commit()
    conn.close()


def get_existing_bvids() -> set:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT bvid FROM videos").fetchall()
    conn.close()
    return {r[0] for r in rows}


def save_video(v: Dict[str, Any], tier_name: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO videos (bvid,title,author,play_count,cover,link,tier_name,crawled_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (v["bvid"], v.get("title", ""), v.get("author", ""),
         int(v.get("play", 0)), v.get("pic", ""), v.get("arcurl", ""),
         tier_name, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def count_videos() -> int:
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    conn.close()
    return n


def count_by_tier() -> Dict[str, int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT tier_name, COUNT(*) FROM videos GROUP BY tier_name").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# ---------- 断点状态 ----------
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_tiers": [], "keyword_pages": {}}


def save_state(state: Dict[str, Any]):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------- 爬虫 ----------
def search_videos(keyword: str, page: int) -> Optional[Dict[str, Any]]:
    """调B站搜索接口，返回单页结果。失败返回 None。"""
    url = (f"{SEARCH_URL}?search_type=video&keyword={quote(keyword)}"
           f"&order=click&page={page}&page_size={PAGE_SIZE}")
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Referer": "https://search.bilibili.com/",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("code") != 0:
            print(f"  [警告] 搜索返回 code={data.get('code')} msg={data.get('message')}")
            return None
        return data.get("data") or {}
    except urllib.error.HTTPError as e:
        print(f"  [HTTP错误] {e.code} {e.reason}")
        return None
    except Exception as e:
        print(f"  [请求异常] {type(e).__name__}: {e}")
        return None


def crawl_tier(tier: Tuple[int, int, str], state: Dict[str, Any],
               existing: set, max_per_tier: int) -> int:
    """爬取一个播放量梯队，返回新增视频数。"""
    lo, hi, name = tier
    print(f"\n{'='*60}")
    print(f"开始爬取梯队: {name} (播放量 {lo}~{hi})")
    print(f"{'='*60}")

    added = 0
    for kw in KEYWORDS:
        if added >= max_per_tier:
            print(f"  已达该梯队上限 {max_per_tier}，停止")
            break
        state_key = f"{name}|{kw}"
        start_page = state["keyword_pages"].get(state_key, 1)
        print(f"\n  关键词「{kw}」从第 {start_page} 页开始")

        for page in range(start_page, MAX_PAGES_PER_KEYWORD + 1):
            if added >= max_per_tier:
                break
            data = search_videos(kw, page)
            if data is None:
                print(f"    第{page}页请求失败，跳过该关键词")
                break
            results = data.get("result") or []
            if not results:
                print(f"    第{page}页无结果，该关键词结束")
                break

            page_added = 0
            for v in results:
                bvid = v.get("bvid", "")
                if not bvid or bvid in existing:
                    continue
                play = int(v.get("play", 0))
                # 分梯队：只收当前梯队的；不在范围的留给后续梯队或跳过
                if not (lo <= play < hi):
                    continue
                save_video(v, name)
                existing.add(bvid)
                added += 1
                page_added += 1
                if added >= max_per_tier:
                    break

            print(f"    第{page}页: 命中 {page_added} 个，累计 {added}")
            state["keyword_pages"][state_key] = page + 1
            save_state(state)

            # 频率控制：随机 sleep
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        # 关键词结束
        state["keyword_pages"][state_key] = MAX_PAGES_PER_KEYWORD + 1
        save_state(state)

    print(f"\n梯队「{name}」完成，新增 {added} 个视频")
    return added


def run_crawl(tiers_filter: Optional[List[str]] = None,
              max_per_tier: int = 500):
    """主爬取流程。tiers_filter: 仅爬指定梯队名列表；None 表示全部。"""
    init_db()
    existing = get_existing_bvids()
    state = load_state()

    print(f"已有视频: {len(existing)} 个")
    print(f"当前统计: {count_by_tier()}")

    total_added = 0
    for tier in TIERS:
        lo, hi, name = tier
        if tiers_filter and name not in tiers_filter:
            continue
        if name in state.get("completed_tiers", []):
            print(f"\n梯队「{name}」已完成，跳过")
            continue
        added = crawl_tier(tier, state, existing, max_per_tier)
        total_added += added
        state.setdefault("completed_tiers", []).append(name)
        save_state(state)
        print(f"\n累计已爬: {count_videos()} 个视频")

    print(f"\n{'='*60}")
    print(f"爬取完成！本次新增 {total_added} 个，总计 {count_videos()} 个视频")
    print(f"各梯队统计: {count_by_tier()}")
    print(f"{'='*60}")


# ---------- 命令行 ----------
def main():
    parser = argparse.ArgumentParser(description="B站 Wota艺 视频元数据爬虫")
    parser.add_argument("--max-per-tier", type=int, default=500,
                        help="每个播放量梯队最多爬取的视频数（默认500）")
    parser.add_argument("--tier", type=str, default=None,
                        help="只爬指定梯队名（如 '1万以上'），默认全部")
    parser.add_argument("--reset-state", action="store_true",
                        help="重置断点状态（不清空数据库，只重新翻页）")
    args = parser.parse_args()

    if args.reset_state and os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)
        print("已重置断点状态")

    tiers_filter = [args.tier] if args.tier else None
    run_crawl(tiers_filter=tiers_filter, max_per_tier=args.max_per_tier)


if __name__ == "__main__":
    main()
