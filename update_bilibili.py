"""
B站视频库每周增量更新脚本
==========================
一键完成：增量爬取新视频 → 增量提取 dHash → 触发 matcher 热重载。

用法：
  python update_bilibili.py              # 默认：新增50个视频
  python update_bilibili.py --max-new 100
  python update_bilibili.py --crawl-only  # 只爬取不提取
  python update_bilibili.py --extract-only # 只提取不爬取

运行后服务器上的 /match-video 会在下次请求时自动检测到库更新并重载。
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))


def main():
    parser = argparse.ArgumentParser(description="B站视频库每周增量更新")
    parser.add_argument("--max-new", type=int, default=50, help="增量爬取最多新增视频数")
    parser.add_argument("--max-pages", type=int, default=5, help="增量爬取每关键词翻页数")
    parser.add_argument("--crawl-only", action="store_true", help="只爬取元数据不提取特征")
    parser.add_argument("--extract-only", action="store_true", help="只提取特征不爬取")
    args = parser.parse_args()

    print("=" * 60)
    print("B站视频库增量更新")
    print("=" * 60)

    # 1. 增量爬取
    if not args.extract_only:
        print("\n[步骤1/3] 增量爬取新视频...")
        import bilibili_crawler as crawler
        added = crawler.run_incremental(max_new=args.max_new, max_pages=args.max_pages)
        print(f"新增 {added} 个视频元数据")
    else:
        print("\n[步骤1/3] 跳过爬取（--extract-only）")

    # 2. 增量提取 dHash
    if not args.crawl_only:
        print("\n[步骤2/3] 增量提取 dHash 特征...")
        import bilibili_feature_builder as builder
        builder.run(limit=10**9)  # run 内部有断点续传，只处理未处理的视频
    else:
        print("\n[步骤2/3] 跳过提取（--crawl-only）")

    # 3. 提示热重载
    print("\n[步骤3/3] 库已更新")
    print("  matcher 会在下次 /match-video 请求时自动检测到变化并重载")
    print("  无需重启服务")

    print("\n" + "=" * 60)
    print("增量更新完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
