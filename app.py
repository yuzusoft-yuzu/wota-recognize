"""
Wota艺 技术识别 - Web 后端
==========================
Flask 服务：接收视频上传 → 光流提取 → 向量检索 → 返回结果
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

from optical_flow_wota import WotaOpticalFlowPipeline
from vector_db_wota import FeatureProjector, MoveRecord, WotaVectorDB

# ---------- Flask 配置 ----------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "wota-艺-secret-" + uuid.uuid4().hex[:16])
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "uploads")
app.config["OUTPUT_FOLDER"] = str(BASE_DIR / "static" / "output")
app.config["DB_PATH"] = os.environ.get("WOTA_DB_PATH", str(BASE_DIR / "wota_db"))

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ---------- 管理员鉴权 ----------
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")  # 生产环境务必修改
ADMIN_TOKEN = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()


def _check_admin_auth(request) -> bool:
    """检查请求是否携带有效的管理员令牌"""
    token = request.headers.get("X-Admin-Token", "")
    if not token:
        token = request.args.get("admin_token", "")
    return hmac.compare_digest(token, ADMIN_TOKEN)


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

# 全局任务状态（简单内存存储）
TASKS: dict = {}


# ===================================================================
# 依赖检测
# ===================================================================
_MISSING_DEPS = []

try:
    import cv2
except ImportError:
    _MISSING_DEPS.append("opencv-python (pip install opencv-python)")

try:
    import numpy as np
except ImportError:
    _MISSING_DEPS.append("numpy (pip install numpy)")

try:
    import scipy
except ImportError:
    _MISSING_DEPS.append("scipy (pip install scipy)")

try:
    import faiss
except ImportError:
    _MISSING_DEPS.append("faiss-cpu (pip install faiss-cpu)")

if _MISSING_DEPS:
    print(f"[警告] 缺少以下依赖，部分功能不可用:")
    for d in _MISSING_DEPS:
        print(f"  - {d}")

# ===================================================================
# 辅助函数
# ===================================================================
def get_db() -> WotaVectorDB | None:
    """加载数据库，不存在则返回 None"""
    meta = app.config["DB_PATH"] + ".meta"
    if os.path.exists(meta):
        return WotaVectorDB.load(app.config["DB_PATH"])
    return None


def process_video(task_id: str, video_path: str, colors: list[str]):
    """
    后台处理视频：
      1. 光流提取
      2. 向量检索
      3. 更新任务状态
    """
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["progress"] = 10

        # ---- Step 1: 光流提取 ----
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
        pipeline = WotaOpticalFlowPipeline(
            video_path=video_path,
            method="farneback",
            color_presets=colors or ["white"],
            mode="light_tracking",
        )
        TASKS[task_id]["progress"] = 30
        pipeline.run(step=1, max_frames=300, output_dir=output_dir)
        TASKS[task_id]["progress"] = 70

        raw_vector = pipeline.get_embedding_snapshot(normalize=True)

        # ---- Step 2: 投影到目标维度 ----
        db = get_db()
        if db and db.dim > len(raw_vector):
            projector = FeatureProjector(input_dim=len(raw_vector), output_dim=db.dim)
            query_vec = projector.project(raw_vector)
        else:
            query_vec = raw_vector.astype(np.float32)

        # ---- Step 3: 向量检索 ----
        TASKS[task_id]["progress"] = 85
        results = []
        if db:
            results = db.search(query_vec, k=5, min_score=0.3)

        # ---- Step 4: 更新结果 ----
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["status"] = "done"
        TASKS[task_id]["result"] = {
            "raw_vector_dim": int(len(raw_vector)),
            "query_vector": query_vec.tolist()[:20],  # 只展示前20维
            "predictions": [
                {
                    "move_name": r["move_name"],
                    "category": r["category"],
                    "confidence": round(r["confidence"] * 100, 1),
                    "confidence_decimal": r["confidence"],
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


# ===================================================================
# 页面路由
# ===================================================================
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


# ===================================================================
# API 路由
# ===================================================================
@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    上传视频并开始识别。
    POST multipart/form-data
      - video: 视频文件
      - colors: (可选) 光棒颜色，逗号分隔，如 "orange,cyan"
    """
    if "video" not in request.files:
        return jsonify({"error": "请上传视频文件"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    # 颜色参数
    colors_raw = request.form.get("colors", "white")
    colors = [c.strip() for c in colors_raw.split(",") if c.strip()]

    # 保存视频
    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
        ext = ".mp4"
    task_id = uuid.uuid4().hex[:8]
    safe_name = f"{task_id}{ext}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(video_path)

    # 启动后台任务
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


@app.route("/api/health")
def api_health():
    """健康检查 & 依赖状态"""
    return jsonify({
        "status": "ok",
        "dependencies": {
            "cv2": "opencv-python" not in " ".join(_MISSING_DEPS),
            "numpy": "numpy" not in " ".join(_MISSING_DEPS),
            "scipy": "scipy" not in " ".join(_MISSING_DEPS),
            "faiss": "faiss" not in " ".join(_MISSING_DEPS),
        },
        "missing": _MISSING_DEPS,
        "db_exists": os.path.exists(app.config["DB_PATH"] + ".meta"),
    })


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
    """
    Web 端入库标准视频。【需要管理员权限】
    """
    if not _check_admin_auth(request):
        return jsonify({"error": "需要管理员权限，请先登录"}), 403

    """
    POST multipart/form-data
      - video: 标准视频文件
      - name: 技术名称
      - category: 技术分类
      - colors: 光棒颜色（逗号分隔）
    """
    if "video" not in request.files:
        return jsonify({"error": "请上传标准视频"}), 400

    file = request.files["video"]
    name = request.form.get("name", "").strip()
    category = request.form.get("category", "").strip()

    if not name:
        return jsonify({"error": "请输入技术名称"}), 400
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    colors_raw = request.form.get("colors", "white")
    colors = [c.strip() for c in colors_raw.split(",") if c.strip()]
    bilibili = request.form.get("bilibili", "").strip()

    # 保存视频
    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"):
        ext = ".mp4"
    file_id = uuid.uuid4().hex[:8]
    safe_name = f"std_{file_id}{ext}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(video_path)

    try:
        # 光流提取
        if "opencv-python" in " ".join(_MISSING_DEPS):
            return jsonify({"error": "缺少 opencv-python，请先运行: pip install opencv-python"}), 500
        if "numpy" in " ".join(_MISSING_DEPS):
            return jsonify({"error": "缺少 numpy，请先运行: pip install numpy"}), 500
        if "scipy" in " ".join(_MISSING_DEPS):
            return jsonify({"error": "缺少 scipy，请先运行: pip install scipy"}), 500
        if "faiss" in " ".join(_MISSING_DEPS):
            return jsonify({"error": "缺少 faiss-cpu，请先运行: pip install faiss-cpu"}), 500

        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], "std_" + file_id)
        pipeline = WotaOpticalFlowPipeline(
            video_path=video_path,
            method="farneback",
            color_presets=colors,
            mode="light_tracking",
        )
        pipeline.run(step=1, max_frames=300, output_dir=output_dir)
        raw_vector = pipeline.get_embedding_snapshot(normalize=True)

        # 入库
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
            source_video=file.filename,
            bilibili=bilibili,
            vector=vec,
        )
        mid = db.insert(record)
        db.save(app.config["DB_PATH"])

        return jsonify({
            "success": True,
            "move_id": mid,
            "move_name": name,
            "category": category,
            "total_count": db.count(),
        })

    except Exception as e:
        return jsonify({"error": f"入库失败: {str(e)}"}), 500


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
if __name__ == "__main__":
    db = get_db()
    if db:
        print(f"数据库已加载: {db.count()} 条技术")
    else:
        print("数据库为空，请通过 Web 界面「数据库管理」录入标准视频")
    if _MISSING_DEPS:
        print(f"缺少 {len(_MISSING_DEPS)} 个依赖，部分功能不可用:")
        for d in _MISSING_DEPS:
            print(f"  > {d}")
    print(f"健康检查: http://127.0.0.1:5000/api/health")
    print(f"访问 http://127.0.0.1:5000\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
