"""
Wota艺 技术识别 - Web 后端
==========================
Flask 服务：接收视频上传 → 光流提取 → 向量检索 → 返回结果

使用延迟加载以降低启动内存开销，后台预加载避免首次请求超时。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sys
import uuid
import json
import shutil
import threading
import importlib.util
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, url_for, session,
)
from werkzeug.utils import secure_filename

# ---------- 项目根目录 ----------
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# ---------- Flask 配置 ----------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "wota-艺-secret-" + uuid.uuid4().hex[:16])
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
app.config["OUTPUT_FOLDER"] = os.environ.get("OUTPUT_FOLDER", str(BASE_DIR / "static" / "output"))
app.config["DB_PATH"] = os.environ.get("WOTA_DB_PATH", str(BASE_DIR / "wota_db"))

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ---------- 管理员鉴权 ----------
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
ADMIN_TOKEN = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()


def _check_admin_auth(request) -> bool:
    """检查请求是否携带有效的管理员令牌"""
    token = request.headers.get("X-Admin-Token", "")
    if not token:
        token = request.args.get("admin_token", "")
    return hmac.compare_digest(token, ADMIN_TOKEN)


# ===================================================================
# 延迟加载：重量级依赖
# ===================================================================
_optical_flow_module = None
_vector_db_module = None
_cached_db = None
_dep_checks = {}


def _lazy_import_optical_flow():
    """延迟加载光流模块（首次调用时才导入 cv2, numpy, scipy 等）"""
    global _optical_flow_module
    if _optical_flow_module is None:
        from optical_flow_wota import WotaOpticalFlowPipeline
        _optical_flow_module = WotaOpticalFlowPipeline
    return _optical_flow_module


def _lazy_import_vector_db():
    """延迟加载向量数据库模块（首次调用时才导入 faiss 等）"""
    global _vector_db_module
    if _vector_db_module is None:
        from vector_db_wota import FeatureProjector, MoveRecord, WotaVectorDB
        _vector_db_module = (FeatureProjector, MoveRecord, WotaVectorDB)
    return _vector_db_module


def _check_dep(name: str) -> bool:
    """轻量级检查依赖是否可用（使用 importlib，不实际加载模块）"""
    if name in _dep_checks:
        return _dep_checks[name]
    _dep_checks[name] = importlib.util.find_spec(name) is not None
    return _dep_checks[name]


def _preload_deps():
    """后台预加载所有重量级依赖，避免首次请求超时"""
    try:
        import cv2
        _dep_checks["cv2"] = True
        print("[预加载] cv2 完成")
    except ImportError:
        _dep_checks["cv2"] = False
        print("[预加载] cv2 失败")

    try:
        import numpy as np
        _dep_checks["numpy"] = True
        print("[预加载] numpy 完成")
    except ImportError:
        _dep_checks["numpy"] = False
        print("[预加载] numpy 失败")

    try:
        import scipy
        _dep_checks["scipy"] = True
        print("[预加载] scipy 完成")
    except ImportError:
        _dep_checks["scipy"] = False
        print("[预加载] scipy 失败")

    try:
        import faiss
        _dep_checks["faiss"] = True
        print("[预加载] faiss 完成")
    except ImportError:
        _dep_checks["faiss"] = False
        print("[预加载] faiss 失败")

    # 预热光流模块和数据库模块
    try:
        _lazy_import_optical_flow()
        print("[预加载] optical_flow_wota 完成")
    except Exception as e:
        print(f"[预加载] optical_flow_wota 失败: {e}")

    try:
        _lazy_import_vector_db()
        print("[预加载] vector_db_wota 完成")
    except Exception as e:
        print(f"[预加载] vector_db_wota 失败: {e}")

    print("[预加载] 所有依赖加载完毕")


# ===================================================================
# 辅助函数
# ===================================================================
def get_db():
    """加载并缓存数据库，不存在则返回 None"""
    global _cached_db
    if _cached_db is not None:
        return _cached_db
    meta = app.config["DB_PATH"] + ".meta"
    if os.path.exists(meta):
        _, _, WotaVectorDB = _lazy_import_vector_db()
        _cached_db = WotaVectorDB.load(app.config["DB_PATH"])
        return _cached_db
    return None


# 全局任务状态
TASKS: dict = {}
DB_TASKS: dict = {}
UPLOAD_SESSIONS: dict = {}  # 分片上传会话


# ===================================================================
# 业务逻辑
# ===================================================================
def preprocess_video(video_path: str) -> str:
    """用 ffmpeg 压缩视频到 480p 30fps，降低内存占用。返回新路径（或原路径）"""
    import subprocess
    compressed_path = video_path + "_compressed.mp4"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", "scale=-2:480,fps=30",
             "-c:v", "libx264", "-crf", "28", "-preset", "fast",
             "-c:a", "aac", "-b:a", "64k",
             compressed_path],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.exists(compressed_path):
            # 替换原文件
            os.replace(compressed_path, video_path)
            return video_path
    except Exception as e:
        print(f"[预处理] ffmpeg 压缩失败: {e}")
    return video_path


def process_video(task_id: str, video_path: str, colors: list[str]):
    """后台处理视频：光流提取 → 向量检索 → 更新任务状态"""
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["progress"] = 5

        # 预处理：压缩视频降低内存
        video_path = preprocess_video(video_path)
        TASKS[task_id]["progress"] = 10

        # 延迟加载光流模块（首次调用时才导入 cv2, numpy, scipy 等）
        WotaOpticalFlowPipeline = _lazy_import_optical_flow()
        TASKS[task_id]["progress"] = 15
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
        pipeline = WotaOpticalFlowPipeline(
            video_path=video_path,
            method="farneback",
            color_presets=colors or ["white"],
            mode="light_tracking",
        )
        TASKS[task_id]["progress"] = 30
        pipeline.run(step=1, max_frames=150, output_dir=output_dir)
        TASKS[task_id]["progress"] = 70

        raw_vector = pipeline.get_embedding_snapshot(normalize=True)

        # numpy 在 lazy import 时已经加载
        import numpy as np

        FeatureProjector, _, _ = _lazy_import_vector_db()
        db = get_db()
        if db and db.dim > len(raw_vector):
            projector = FeatureProjector(input_dim=len(raw_vector), output_dim=db.dim)
            query_vec = projector.project(raw_vector)
        else:
            query_vec = raw_vector.astype(np.float32)

        TASKS[task_id]["progress"] = 85
        results = []
        if db:
            results = db.search(query_vec, k=5, min_score=0.3)

        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["status"] = "done"
        TASKS[task_id]["result"] = {
            "raw_vector_dim": int(len(raw_vector)),
            "query_vector": query_vec.tolist()[:20],
            "predictions": [
                {
                    "move_name": r["move_name"],
                    "category": r["category"],
                    "confidence": round(r["confidence"] * 100, 1),
                }
                for r in results
            ],
            "top_match": results[0]["move_name"] if results else "未识别",
            "top_confidence": round(results[0]["confidence"] * 100, 1) if results else 0,
            "has_db": db is not None,
            "output_files": {
                "trajectories": f"/static/output/{task_id}/trajectories.png",
                "light_arc": f"/static/output/{task_id}/light_arc.png",
                "optical_flow_video": f"/static/output/{task_id}/optical_flow.mp4",
                "centroid_video": f"/static/output/{task_id}/centroid_tracking.mp4",
            },
        }

    except Exception as e:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)


def process_db_add(task_id: str, video_path: str, name: str, category: str,
                   colors: list[str], bilibili: str, source_filename: str):
    """后台入库处理：光流提取 → 向量入库 → 保存数据库"""
    try:
        DB_TASKS[task_id]["status"] = "processing"
        DB_TASKS[task_id]["progress"] = 10

        # 预处理：压缩视频降低内存
        video_path = preprocess_video(video_path)
        DB_TASKS[task_id]["progress"] = 20

        WotaOpticalFlowPipeline = _lazy_import_optical_flow()
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], "std_" + task_id)
        pipeline = WotaOpticalFlowPipeline(
            video_path=video_path,
            method="farneback",
            color_presets=colors,
            mode="light_tracking",
        )
        DB_TASKS[task_id]["progress"] = 40
        pipeline.run(step=1, max_frames=150, output_dir=output_dir)
        DB_TASKS[task_id]["progress"] = 70
        raw_vector = pipeline.get_embedding_snapshot(normalize=True)

        import numpy as np
        FeatureProjector, MoveRecord, WotaVectorDB = _lazy_import_vector_db()

        db = get_db()
        if db is None:
            db = WotaVectorDB(dim=512)
        if db.dim > len(raw_vector):
            projector = FeatureProjector(input_dim=len(raw_vector), output_dim=db.dim)
            vec = projector.project(raw_vector)
        else:
            vec = raw_vector.astype(np.float32)

        record = MoveRecord(
            move_name=name,
            category=category,
            source_video=source_filename,
            bilibili=bilibili,
            vector=vec,
        )
        mid = db.insert(record)
        db.save(app.config["DB_PATH"])

        # 刷新缓存
        global _cached_db
        _cached_db = db

        DB_TASKS[task_id]["progress"] = 100
        DB_TASKS[task_id]["status"] = "done"
        DB_TASKS[task_id]["result"] = {
            "success": True,
            "move_id": mid,
            "move_name": name,
            "category": category,
            "total_count": db.count(),
        }

    except Exception as e:
        DB_TASKS[task_id]["status"] = "error"
        DB_TASKS[task_id]["error"] = str(e)


# ===================================================================
# 路由
# ===================================================================
@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    """管理员登录，返回令牌"""
    data = request.get_json() or {}
    password = data.get("password", "")
    if password != ADMIN_PASSWORD:
        return jsonify({"error": "密码错误"}), 401
    return jsonify({"token": ADMIN_TOKEN, "success": True})


@app.route("/api/auth/check", methods=["GET"])
def api_auth_check():
    """检查当前令牌是否有效"""
    if _check_admin_auth(request):
        return jsonify({"valid": True})
    return jsonify({"valid": False})


@app.route("/api/health")
def api_health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "service": "wota-recognize",
        "db_exists": os.path.exists(app.config["DB_PATH"] + ".meta"),
    })


@app.route("/")
def index():
    """主页"""
    db = get_db()
    db_info = {"exists": False, "count": 0, "moves": []}
    if db:
        db_info["exists"] = True
        db_info["count"] = db.count()
        db_info["moves"] = db.list_all()
    return render_template("index.html", db_info=db_info)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """上传视频并开始识别"""
    if "video" not in request.files:
        return jsonify({"error": "请上传视频文件"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    colors_raw = request.form.get("colors", "white")
    colors = [c.strip() for c in colors_raw.split(",") if c.strip()]

    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
        ext = ".mp4"
    task_id = uuid.uuid4().hex[:8]
    safe_name = f"{task_id}{ext}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(video_path)

    TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": 0,
        "video_name": file.filename,
        "created_at": datetime.now().isoformat(),
    }

    thread = threading.Thread(
        target=process_video,
        args=(task_id, video_path, colors),
        daemon=True,
    )
    thread.start()

    return jsonify({"task_id": task_id, "status": "queued"})


@app.route("/api/task/<task_id>")
def api_task_status(task_id: str):
    """轮询任务状态"""
    task = TASKS.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404

    resp = {
        "task_id": task["task_id"],
        "status": task["status"],
        "progress": task["progress"],
        "video_name": task.get("video_name", ""),
    }
    if task["status"] == "done":
        resp["result"] = task["result"]
    elif task["status"] == "error":
        resp["error"] = task.get("error", "未知错误")

    return jsonify(resp)


@app.route("/api/db/info")
def api_db_info():
    """数据库信息"""
    db = get_db()
    if not db:
        return jsonify({"exists": False, "count": 0, "moves": []})
    return jsonify({
        "exists": True,
        "count": db.count(),
        "dim": db.dim,
        "moves": db.list_all(),
    })


@app.route("/api/db/add", methods=["POST"])
def api_db_add():
    """Web 端入库标准视频（异步）。【需要管理员权限】"""
    if not _check_admin_auth(request):
        return jsonify({"error": "需要管理员权限，请先登录"}), 403

    if "video" not in request.files:
        return jsonify({"error": "请上传标准视频"}), 400

    file = request.files["video"]
    name = request.form.get("name", "").strip()
    category = request.form.get("category", "").strip()

    if not name:
        return jsonify({"error": "请输入技术名称"}), 400
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 依赖检查（延迟方式）
    if not _check_dep("cv2"):
        return jsonify({"error": "缺少 opencv-python"}), 500
    if not _check_dep("numpy"):
        return jsonify({"error": "缺少 numpy"}), 500
    if not _check_dep("scipy"):
        return jsonify({"error": "缺少 scipy"}), 500
    if not _check_dep("faiss"):
        return jsonify({"error": "缺少 faiss-cpu"}), 500

    colors_raw = request.form.get("colors", "white")
    colors = [c.strip() for c in colors_raw.split(",") if c.strip()]
    bilibili = request.form.get("bilibili", "").strip()

    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
        ext = ".mp4"
    task_id = uuid.uuid4().hex[:8]
    safe_name = f"std_{task_id}{ext}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(video_path)

    DB_TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": 5,
        "move_name": name,
        "created_at": datetime.now().isoformat(),
    }

    thread = threading.Thread(
        target=process_db_add,
        args=(task_id, video_path, name, category, colors, bilibili, file.filename),
        daemon=True,
    )
    thread.start()

    return jsonify({"task_id": task_id, "status": "queued"})


@app.route("/api/db/add/status/<task_id>")
def api_db_add_status(task_id: str):
    """查询入库任务状态。【需要管理员权限】"""
    if not _check_admin_auth(request):
        return jsonify({"error": "需要管理员权限"}), 403
    task = DB_TASKS.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404

    resp = {
        "task_id": task["task_id"],
        "status": task["status"],
        "progress": task["progress"],
        "move_name": task.get("move_name", ""),
    }
    if task["status"] == "done":
        resp["result"] = task["result"]
    elif task["status"] == "error":
        resp["error"] = task.get("error", "未知错误")
    return jsonify(resp)


@app.route("/api/db/delete", methods=["POST"])
def api_db_delete():
    """Web 端删除技术。【需要管理员权限】"""
    if not _check_admin_auth(request):
        return jsonify({"error": "需要管理员权限，请先登录"}), 403
    data = request.get_json()
    move_id = data.get("move_id", "").strip()
    if not move_id:
        return jsonify({"error": "缺少 move_id"}), 400

    db = get_db()
    if not db:
        return jsonify({"error": "数据库不存在"}), 404

    ok = db.delete(move_id)
    if ok:
        db.save(app.config["DB_PATH"])
        return jsonify({"success": True, "total_count": db.count()})
    return jsonify({"error": "技术不存在"}), 404


@app.route("/api/db/update", methods=["POST"])
def api_db_update():
    """修改技术详情（名称/分类/链接）。【需要管理员权限】"""
    if not _check_admin_auth(request):
        return jsonify({"error": "需要管理员权限，请先登录"}), 403
    data = request.get_json()
    move_id = data.get("move_id", "").strip()
    if not move_id:
        return jsonify({"error": "缺少 move_id"}), 400

    db = get_db()
    if not db:
        return jsonify({"error": "数据库不存在"}), 404

    ok = db.update_move(
        move_id,
        move_name=data.get("move_name", "").strip() or None,
        category=data.get("category", "").strip() if "category" in data else None,
        bilibili=data.get("bilibili", "").strip() if "bilibili" in data else None,
    )
    if ok:
        db.save(app.config["DB_PATH"])
        return jsonify({"success": True, "total_count": db.count()})
    return jsonify({"error": "技术不存在"}), 404


# ===================================================================
# 静态文件
# ===================================================================
@app.route("/static/output/<task_id>/<filename>")
def serve_output(task_id: str, filename: str):
    """提供处理结果的静态文件"""
    folder = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
    return send_from_directory(folder, filename)


# ===================================================================
# 启动
# ===================================================================
# 分片上传 API（绕过代理超时限制）
# ===================================================================
@app.route("/api/upload/init", methods=["POST"])
def api_upload_init():
    """初始化分片上传会话"""
    data = request.get_json() or {}
    filename = data.get("filename", "")
    total_size = data.get("total_size", 0)
    chunk_size = data.get("chunk_size", 2 * 1024 * 1024)  # 默认 2MB
    total_chunks = data.get("total_chunks", 0)

    if not filename:
        return jsonify({"error": "缺少 filename"}), 400

    upload_id = uuid.uuid4().hex[:12]
    ext = Path(filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
        ext = ".mp4"

    UPLOAD_SESSIONS[upload_id] = {
        "upload_id": upload_id,
        "filename": filename,
        "total_size": total_size,
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
        "ext": ext,
        "chunks_received": set(),
        "created_at": datetime.now().isoformat(),
    }

    return jsonify({"upload_id": upload_id, "chunk_size": chunk_size})


@app.route("/api/upload/chunk", methods=["POST"])
def api_upload_chunk():
    """上传单个分片"""
    upload_id = request.form.get("upload_id", "")
    chunk_index = int(request.form.get("chunk_index", 0))

    if upload_id not in UPLOAD_SESSIONS:
        return jsonify({"error": "上传会话不存在"}), 404

    if "chunk" not in request.files:
        return jsonify({"error": "缺少 chunk 文件"}), 400

    session = UPLOAD_SESSIONS[upload_id]
    chunk = request.files["chunk"]

    # 保存分片
    chunk_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"chunks_{upload_id}")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:06d}")
    chunk.save(chunk_path)

    session["chunks_received"].add(chunk_index)

    return jsonify({
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "received": len(session["chunks_received"]),
        "total": session["total_chunks"],
    })


@app.route("/api/upload/merge", methods=["POST"])
def api_upload_merge():
    """合并所有分片并触发入库/识别（立即返回 task_id，后台合并）"""
    data = request.get_json() or {}
    upload_id = data.get("upload_id", "")
    action = data.get("action", "recognize")  # recognize | db_add

    if upload_id not in UPLOAD_SESSIONS:
        return jsonify({"error": "上传会话不存在"}), 404

    session = UPLOAD_SESSIONS[upload_id]

    # 检查所有分片是否就绪
    expected = set(range(session["total_chunks"]))
    received = session["chunks_received"]
    missing = expected - received
    if missing:
        return jsonify({"error": f"还有 {len(missing)} 个分片未上传"}), 400

    # 权限/参数检查（在返回前完成）
    if action == "db_add":
        name = data.get("name", "").strip()
        if not _check_admin_auth(request):
            return jsonify({"error": "需要管理员权限"}), 403
        if not name:
            return jsonify({"error": "请输入技术名称"}), 400
        category = data.get("category", "").strip()
        colors_raw = data.get("colors", "white")
        colors = [c.strip() for c in colors_raw.split(",") if c.strip()]
        bilibili = data.get("bilibili", "").strip()

        task_id = uuid.uuid4().hex[:8]
        DB_TASKS[task_id] = {
            "task_id": task_id,
            "status": "merging",
            "progress": 1,
            "move_name": name,
            "created_at": datetime.now().isoformat(),
        }
        # 后台线程：合并文件 → 处理入库
        thread = threading.Thread(
            target=_merge_and_process,
            args=(upload_id, task_id, "db_add", {
                "name": name, "category": category,
                "colors": colors, "bilibili": bilibili,
                "filename": session["filename"],
            }),
            daemon=True,
        )
        thread.start()
        return jsonify({"task_id": task_id, "status": "merging"})

    else:  # recognize
        colors_raw = data.get("colors", "white")
        colors = [c.strip() for c in colors_raw.split(",") if c.strip()]
        task_id = uuid.uuid4().hex[:8]
        TASKS[task_id] = {
            "task_id": task_id,
            "status": "merging",
            "progress": 1,
            "video_name": session["filename"],
            "created_at": datetime.now().isoformat(),
        }
        thread = threading.Thread(
            target=_merge_and_process,
            args=(upload_id, task_id, "recognize", {
                "colors": colors, "filename": session["filename"],
            }),
            daemon=True,
        )
        thread.start()
        return jsonify({"task_id": task_id, "status": "merging"})


def _merge_and_process(upload_id: str, task_id: str, action: str, params: dict):
    """后台：合并分片文件 → 触发处理"""
    try:
        session = UPLOAD_SESSIONS.get(upload_id)
        if not session:
            _set_task_error(task_id, action, "上传会话已失效")
            return

        chunk_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"chunks_{upload_id}")
        ext = session["ext"]
        safe_name = f"{upload_id}{ext}"
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

        # 合并文件
        _set_task_status(task_id, action, "merging", 3)
        with open(video_path, "wb") as out_f:
            for i in range(session["total_chunks"]):
                chunk_path = os.path.join(chunk_dir, f"chunk_{i:06d}")
                with open(chunk_path, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)

        # 清理分片
        shutil.rmtree(chunk_dir, ignore_errors=True)
        del UPLOAD_SESSIONS[upload_id]

        _set_task_status(task_id, action, "queued", 5)

        # 触发处理
        if action == "db_add":
            process_db_add(task_id, video_path, params["name"], params["category"],
                           params["colors"], params["bilibili"], params["filename"])
        else:
            process_video(task_id, video_path, params["colors"])

    except Exception as e:
        _set_task_error(task_id, action, f"合并失败: {e}")


def _set_task_status(task_id: str, action: str, status: str, progress: int):
    """更新任务状态"""
    store = DB_TASKS if action == "db_add" else TASKS
    if task_id in store:
        store[task_id]["status"] = status
        store[task_id]["progress"] = progress


def _set_task_error(task_id: str, action: str, error: str):
    """设置任务错误"""
    store = DB_TASKS if action == "db_add" else TASKS
    if task_id in store:
        store[task_id]["status"] = "error"
        store[task_id]["error"] = error


# ===================================================================
if __name__ == "__main__":
    print("WOTA 服务启动中（延迟加载模式）...")
    print(f"健康检查: http://127.0.0.1:5000/api/health")
    print(f"访问 http://127.0.0.1:5000\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
