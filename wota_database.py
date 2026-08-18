"""
Wota艺 技术数据库模块 (骨光融合版)
==================================
基于 SQLite 的技术库，存储：
  - 技名 (move_name)        —— 管理员标注，作为主标签
  - 日语名 (japanese_name)
  - 分类   (category)
  - B站链接 (bilibili)
  - 描述   (description)
  - 来源视频 (source_video)
  - 逐帧特征序列 (feature_sequence) —— JSON 文本，供 DTW 时序比对
  - 帧数 / 时长 / 时间戳

同时管理管理员账号 (账号 + 密码哈希)。

设计要点：
  - 技术总览的技名均来自管理员上传的视频，初始为空。
  - 支持搜索引擎 (按技名/日语名/分类模糊检索)。
  - 管理员可修改详细信息 (B站链接、日语名等)。
  - 逐帧特征序列以 JSON 存储，支持变长，便于 DTW。
"""

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import secrets
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any


# ===================================================================
# 密码哈希 (盐 + sha256，轻量级，单机部署足够)
# ===================================================================
def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(8)
    h = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return f"{salt}${h}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$", 1)
    except ValueError:
        return False
    calc = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return secrets.compare_digest(calc, h)


# ===================================================================
# 技术数据库
# ===================================================================
class WotaDatabase:
    """SQLite 技术库 + 管理员账号管理。线程安全（每调用一个连接）。"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS techniques (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        move_id         TEXT UNIQUE NOT NULL,
        move_name       TEXT NOT NULL,
        japanese_name   TEXT DEFAULT '',
        category        TEXT DEFAULT '',
        bilibili        TEXT DEFAULT '',
        description     TEXT DEFAULT '',
        source_video    TEXT DEFAULT '',
        feature_sequence TEXT,        -- JSON: [[...per-frame vector...], ...]
        frame_count     INTEGER DEFAULT 0,
        duration        REAL DEFAULT 0,
        created_at      TEXT,
        updated_at      TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_move_name ON techniques(move_name);
    CREATE TABLE IF NOT EXISTS admin_users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at    TEXT
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ---------- 连接 ----------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            conn = self._connect()
            try:
                conn.executescript(self.SCHEMA)
                self._seed_admin(conn)
                conn.commit()
            finally:
                conn.close()

    # ---------- 管理员 ----------
    def _seed_admin(self, conn: sqlite3.Connection):
        admin_user = os.environ.get("ADMIN_USER", "admin")
        admin_pwd = os.environ.get("ADMIN_PASSWORD", "admin123")
        row = conn.execute(
            "SELECT id FROM admin_users WHERE username=?", (admin_user,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO admin_users (username, password_hash, created_at) VALUES (?,?,?)",
                (admin_user, hash_password(admin_pwd), datetime.now().isoformat()),
            )

    def verify_admin(self, username: str, password: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT password_hash FROM admin_users WHERE username=?", (username,)
            ).fetchone()
            if row is None:
                return False
            return verify_password(password, row["password_hash"])
        finally:
            conn.close()

    def admin_exists(self, username: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM admin_users WHERE username=?", (username,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    # ---------- 技术增删改查 ----------
    @staticmethod
    def _new_move_id(name: str) -> str:
        raw = f"{name}_{secrets.token_hex(4)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def add_technique(
        self,
        move_name: str,
        feature_sequence: List[List[float]],
        japanese_name: str = "",
        category: str = "",
        bilibili: str = "",
        description: str = "",
        source_video: str = "",
        frame_count: int = 0,
        duration: float = 0.0,
    ) -> str:
        """新增一条技术记录，返回 move_id。技名即管理员标注名。"""
        move_name = move_name.strip()
        if not move_name:
            raise ValueError("技名不能为空")
        move_id = self._new_move_id(move_name)
        now = datetime.now().isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO techniques
                       (move_id, move_name, japanese_name, category, bilibili,
                        description, source_video, feature_sequence, frame_count,
                        duration, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        move_id, move_name, japanese_name.strip(), category.strip(),
                        bilibili.strip(), description.strip(), source_video,
                        json.dumps(feature_sequence, ensure_ascii=False),
                        int(frame_count), float(duration), now, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return move_id

    def update_technique(self, move_id: str, fields: Dict[str, Any]) -> bool:
        """管理员修改详细信息。仅允许更新以下字段。"""
        allowed = {
            "move_name", "japanese_name", "category", "bilibili",
            "description", "source_video",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        if "move_name" in updates:
            updates["move_name"] = str(updates["move_name"]).strip()
            if not updates["move_name"]:
                del updates["move_name"]
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        params = list(updates.values()) + [move_id]
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"UPDATE techniques SET {set_clause} WHERE move_id=?", params
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def delete_technique(self, move_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM techniques WHERE move_id=?", (move_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_technique(self, move_id: str) -> Optional[Dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM techniques WHERE move_id=?", (move_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        d = dict(row)
        return d

    def list_techniques(self) -> List[Dict]:
        """列出所有技术（不含特征序列，减少传输量）。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT id, move_id, move_name, japanese_name, category,
                          bilibili, description, source_video, frame_count,
                          duration, created_at, updated_at
                   FROM techniques ORDER BY id DESC"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search_techniques(self, keyword: str) -> List[Dict]:
        """搜索引擎：按技名/日语名/分类模糊检索。"""
        kw = f"%{keyword.strip()}%"
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT id, move_id, move_name, japanese_name, category,
                          bilibili, description, source_video, frame_count,
                          duration, created_at, updated_at
                   FROM techniques
                   WHERE move_name LIKE ? OR japanese_name LIKE ?
                         OR category LIKE ? OR description LIKE ?
                   ORDER BY id DESC""",
                (kw, kw, kw, kw),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_feature_sequence(self, move_id: str) -> Optional[List[List[float]]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT feature_sequence FROM techniques WHERE move_id=?",
                (move_id,),
            ).fetchone()
            if row is None or not row["feature_sequence"]:
                return None
            return json.loads(row["feature_sequence"])
        finally:
            conn.close()

    def iter_techniques_with_features(self):
        """迭代所有 (metadata, feature_sequence)，供识别器遍历比对。"""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT move_id, move_name, japanese_name, category, bilibili,
                          description, feature_sequence, frame_count, duration
                   FROM techniques"""
            ).fetchall()
            for r in rows:
                seq = json.loads(r["feature_sequence"]) if r["feature_sequence"] else []
                meta = {
                    "move_id": r["move_id"],
                    "move_name": r["move_name"],
                    "japanese_name": r["japanese_name"],
                    "category": r["category"],
                    "bilibili": r["bilibili"],
                    "description": r["description"],
                    "frame_count": r["frame_count"],
                    "duration": r["duration"],
                }
                yield meta, seq
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS c FROM techniques").fetchone()
            return int(row["c"])
        finally:
            conn.close()


# ===================================================================
# 命令行：初始化 / 列表 / 搜索
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wota艺 技术数据库管理")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="初始化数据库")
    p_init.add_argument("-d", "--db", default="./wota_tech.db")

    p_list = sub.add_parser("list", help="列出所有技术")
    p_list.add_argument("-d", "--db", default="./wota_tech.db")

    p_search = sub.add_parser("search", help="搜索技术")
    p_search.add_argument("keyword")
    p_search.add_argument("-d", "--db", default="./wota_tech.db")

    args = parser.parse_args()
    db = WotaDatabase(args.db)

    if args.cmd == "init":
        print(f"[Init] 数据库已就绪: {args.db}")
        print(f"[Init] 默认管理员: {os.environ.get('ADMIN_USER','admin')} / "
              f"{os.environ.get('ADMIN_PASSWORD','admin123')} (可通过环境变量覆盖)")
    elif args.cmd == "list":
        items = db.list_techniques()
        print(f"\n数据库: {len(items)} 条技术")
        print("-" * 50)
        for it in items:
            print(f"  [{it['move_id']}] {it['move_name']}  ({it['category']})")
            if it["japanese_name"]:
                print(f"       日语名: {it['japanese_name']}")
            if it["bilibili"]:
                print(f"       B站: {it['bilibili']}")
    elif args.cmd == "search":
        items = db.search_techniques(args.keyword)
        print(f"搜索 '{args.keyword}': 命中 {len(items)} 条")
        for it in items:
            print(f"  [{it['move_id']}] {it['move_name']}  ({it['category']})")
    else:
        parser.print_help()
