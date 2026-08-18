"""
Wota艺 动作识别网站 - Web 后端 (骨光融合版)
=============================================
三板块多页面架构：
  /            门户首页（三个板块入口）
  /recognize   动作识别（用户上传视频 → 比对数据库 → 返回技名 + 匹配度）
  /techniques  技术总览（搜索引擎 + 技名详情；管理员可修改）
  /admin       管理员界面（账号密码登录 → 上传标准动作视频并标注技名 → 入库）

核心算法见 skeleton_light_fusion.py（骨光融合 + DTW），数据层见 wota_database.py。
"""

from __future__ import annotations

import os
import sys
import uuid
import threading
from pathlib import Path
from datetime import datetime

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, session, redirect, url_for,
)
from werkzeug.utils import secure_filename

# ---------- 项目根目录 ----------
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# ---------- Flask 配置 ----------
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))
app.secret_key = os.environ.get("SECRET_KEY", "wota-艺-secret-" + uuid.uuid4().hex[:16])
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024          # 60MB 上限
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "uploads")
app.config["OUTPUT_FOLDER"] = str(BASE_DIR / "static" / "output")
app.config["DB_PATH"] = os.environ.get("WOTA_DB_PATH", str(BASE_DIR / "wota_tech.db"))
app.config["MAX_STANDARD_MB"] = 50     # 标准动作视频 ≤ 50MB
app.config["MAX_STANDARD_SEC"] = 15    # 标准动作视频时长上限（建议 ~10s，留余量）

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# ===================================================================
# 延迟加载：重量级依赖
# ===================================================================
_fusion_module = None
_db_instance = None
_dep_checks: dict = {}


def _lazy_fusion():
    global _fusion_module
    if _fusion_module is None:
        from skeleton_light_fusion import FusionPipeline, Recognizer, DTWMatcher, _HAS_MP
        _fusion_module = (FusionPipeline, Recognizer, DTWMatcher, _HAS_MP)
    return _fusion_module


def get_db():
    global _db_instance
    if _db_instance is None:
        from wota_database import WotaDatabase
        _db_instance = WotaDatabase(app.config["DB_PATH"])
    return _db_instance


def _check_dep(name: str) -> bool:
    if name in _dep_checks:
        return _dep_checks[name]
    import importlib.util
    _dep_checks[name] = importlib.util.find_spec(name) is not None
    return _dep_checks[name]


def _preload_deps():
    for n in ("cv2", "numpy", "scipy"):
        try:
            __import__(n)
            _dep_checks[n] = True
        except ImportError:
            _dep_checks[n] = False
    try:
        _lazy_fusion()
    except Exception as e:
        print(f"[预加载] fusion 模块失败: {e}")
    try:
        get_db()
        print(f"[预加载] 数据库就绪: {app.config['DB_PATH']} (技术数: {get_db().count()})")
    except Exception as e:
        print(f"[预加载] 数据库失败: {e}")
    print("[预加载] 完成")


# ===================================================================
# 鉴权
# ===================================================================
def is_admin() -> bool:
    return bool(session.get("admin"))


def _valid_video_ext(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"):
        return ".mp4"
    return ext


def _save_upload(file, prefix: str) -> str:
    ext = _valid_video_ext(file.filename or "")
    safe = f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], safe)
    file.save(path)
    return path


def _file_size_ok(path: str, max_mb: int) -> bool:
    try:
        return os.path.getsize(path) <= max_mb * 1024 * 1024
    except OSError:
        return False


# 全局任务状态（内存）
TASKS: dict = {}      # 识别任务
STD_TASKS: dict = {}  # 标准化(入库)任务


# ===================================================================
# 后台处理
# ===================================================================
def _progress(store, task_id, status, progress):
    if task_id in store:
        store[task_id]["status"] = status
        store[task_id]["progress"] = progress


def _progress_err(store, task_id, err):
    if task_id in store:
        store[task_id]["status"] = "error"
        store[task_id]["error"] = err


def _extract_feature_sequence(video_path: str, task_id: str, store: dict):
    """共用：骨光融合特征提取，带进度上报。"""
    FusionPipeline, _, _, _ = _lazy_fusion()
    _progress(store, task_id, "extracting", 25)
    pipe = FusionPipeline(step=2, max_frames=150, n_workers=4, use_skeleton=True)
    res = pipe.extract(video_path)
    _progress(store, task_id, "finalizing", 75)
    # 预览图
    preview_path = os.path.join(app.config["OUTPUT_FOLDER"], task_id, "preview.png")
    pipe.render_preview(video_path, preview_path)
    preview_url = f"/output/{task_id}/preview.png" if os.path.exists(preview_path) else None
    return res, preview_url, pipe.use_skeleton


def process_recognize(task_id: str, video_path: str):
    """动作识别：提取查询特征 → 与库内所有技术 DTW 比对 → 排名。"""
    store = TASKS
    try:
        _progress(store, task_id, "processing", 10)
        # 依赖检查
        for dep in ("cv2", "numpy"):
            if not _check_dep(dep):
                raise RuntimeError(f"缺少依赖 {dep}")
        res, preview_url, used_sk = _extract_feature_sequence(video_path, task_id, store)
        query_seq = res["feature_sequence"]

        _progress(store, task_id, "matching", 80)
        db = get_db()
        _, Recognizer, _, _ = _lazy_fusion()
        candidates = list(db.iter_techniques_with_features())
        recognizer = Recognizer(n_slices=3, min_score=0.25, top_k=5)
        predictions = recognizer.recognize(query_seq, candidates)

        _progress(store, task_id, "done", 100)
        top = predictions[0] if predictions else None
        store[task_id]["result"] = {
            "has_db": db.count() > 0,
            "db_count": db.count(),
            "used_skeleton": used_sk,
            "query_frames": res["frame_count"],
            "duration": round(res["duration"], 2),
            "top_match": top["move_name"] if top else "未识别",
            "top_japanese": top["japanese_name"] if top else "",
            "top_match_percent": top["match"] if top else 0.0,
            "top_bilibili": top["bilibili"] if top else "",
            "predictions": [
                {
                    "move_id": p["move_id"],
                    "move_name": p["move_name"],
                    "japanese_name": p["japanese_name"],
                    "category": p["category"],
                    "bilibili": p["bilibili"],
                    "match": p["match"],
                    "full_sim": p["full_sim"],
                    "vote_fraction": p["vote_fraction"],
                }
                for p in predictions
            ],
            "preview_url": preview_url,
        }
    except Exception as e:
        _progress_err(store, task_id, str(e))


def process_standardize(task_id: str, video_path: str, fields: dict, src_name: str):
    """管理员标准化：提取特征 → 以管理员标注的技名入库。"""
    store = STD_TASKS
    try:
        _progress(store, task_id, "processing", 10)
        for dep in ("cv2", "numpy"):
            if not _check_dep(dep):
                raise RuntimeError(f"缺少依赖 {dep}")
        res, preview_url, used_sk = _extract_feature_sequence(video_path, task_id, store)
        seq = res["feature_sequence"]

        _progress(store, task_id, "storing", 88)
        db = get_db()
        move_id = db.add_technique(
            move_name=fields["name"],
            feature_sequence=seq,
            japanese_name=fields.get("japanese_name", ""),
            category=fields.get("category", ""),
            bilibili=fields.get("bilibili", ""),
            description=fields.get("description", ""),
            source_video=src_name,
            frame_count=res["frame_count"],
            duration=res["duration"],
        )
        _progress(store, task_id, "done", 100)
        store[task_id]["result"] = {
            "success": True,
            "move_id": move_id,
            "move_name": fields["name"],
            "total_count": db.count(),
            "used_skeleton": used_sk,
            "query_frames": res["frame_count"],
            "duration": round(res["duration"], 2),
            "preview_url": preview_url,
        }
    except Exception as e:
        _progress_err(store, task_id, str(e))


# ===================================================================
# 页面路由
# ===================================================================
@app.route("/")
def portal():
    return render_template("portal.html", active="home")


@app.route("/recognize")
def page_recognize():
    return render_template("recognize.html", active="recognize")


@app.route("/techniques")
def page_techniques():
    return render_template("techniques.html", active="techniques")


@app.route("/admin")
def page_admin():
    return render_template("admin.html", active="admin", is_admin=is_admin())


# ===================================================================
# 鉴权 API
# ===================================================================
@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"success": False, "error": "请输入账号和密码"}), 400
    db = get_db()
    if db.verify_admin(username, password):
        session["admin"] = True
        session["admin_user"] = username
        return jsonify({"success": True, "username": username})
    return jsonify({"success": False, "error": "账号或密码错误"}), 401


@app.route("/api/auth/check", methods=["GET"])
def api_auth_check():
    return jsonify({"authed": is_admin(), "username": session.get("admin_user", "")})


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.pop("admin", None)
    session.pop("admin_user", None)
    return jsonify({"success": True})


# ===================================================================
# 技术总览 API（用户可搜索/查看；管理员可改/删）
# ===================================================================
@app.route("/api/techniques")
def api_techniques_list():
    q = (request.args.get("q") or "").strip()
    db = get_db()
    items = db.search_techniques(q) if q else db.list_techniques()
    return jsonify({"count": len(items), "items": items})


@app.route("/api/techniques/<move_id>")
def api_technique_detail(move_id: str):
    db = get_db()
    it = db.get_technique(move_id)
    if not it:
        return jsonify({"error": "技术不存在"}), 404
    # 不向前端暴露 feature_sequence（体积大）
    it.pop("feature_sequence", None)
    it["is_admin"] = is_admin()
    return jsonify(it)


@app.route("/api/techniques/<move_id>", methods=["PUT"])
def api_technique_update(move_id: str):
    if not is_admin():
        return jsonify({"error": "需要管理员权限"}), 403
    data = request.get_json(silent=True) or {}
    allowed = {}
    for k in ("move_name", "japanese_name", "category", "bilibili",
              "description", "source_video"):
        if k in data:
            allowed[k] = data[k]
    db = get_db()
    ok = db.update_technique(move_id, allowed)
    if not ok:
        return jsonify({"error": "更新失败（技术不存在或无有效字段）"}), 404
    return jsonify({"success": True})


@app.route("/api/techniques/<move_id>", methods=["DELETE"])
def api_technique_delete(move_id: str):
    if not is_admin():
        return jsonify({"error": "需要管理员权限"}), 403
    db = get_db()
    ok = db.delete_technique(move_id)
    if not ok:
        return jsonify({"error": "技术不存在"}), 404
    return jsonify({"success": True, "total_count": db.count()})


# ===================================================================
# 动作识别 API（用户）
# ===================================================================
@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    if "video" not in request.files:
        return jsonify({"error": "请上传视频文件"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400
    path = _save_upload(file, "rec")
    if not _file_size_ok(path, app.config["MAX_STANDARD_MB"]):
        os.remove(path)
        return jsonify({"error": f"视频过大，请压缩到 {app.config['MAX_STANDARD_MB']}MB 以内"}), 413

    task_id = uuid.uuid4().hex[:10]
    TASKS[task_id] = {
        "task_id": task_id, "status": "queued", "progress": 2,
        "video_name": file.filename, "created_at": datetime.now().isoformat(),
    }
    threading.Thread(target=process_recognize, args=(task_id, path), daemon=True).start()
    return jsonify({"task_id": task_id, "status": "queued"})


@app.route("/api/recognize/<task_id>")
def api_recognize_status(task_id: str):
    t = TASKS.get(task_id)
    if not t:
        return jsonify({"error": "任务不存在"}), 404
    resp = {"task_id": t["task_id"], "status": t["status"],
            "progress": t.get("progress", 0), "video_name": t.get("video_name", "")}
    if t["status"] == "done":
        resp["result"] = t["result"]
    elif t["status"] == "error":
        resp["error"] = t.get("error", "未知错误")
    return jsonify(resp)


# ===================================================================
# 管理员标准化 API（上传标准动作视频 → 标注技名 → 入库）
# ===================================================================
@app.route("/api/admin/upload", methods=["POST"])
def api_admin_upload():
    if not is_admin():
        return jsonify({"error": "需要管理员权限，请先登录"}), 403
    if "video" not in request.files:
        return jsonify({"error": "请上传标准动作视频"}), 400
    file = request.files["video"]
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "请输入技名"}), 400
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    path = _save_upload(file, "std")
    if not _file_size_ok(path, app.config["MAX_STANDARD_MB"]):
        os.remove(path)
        return jsonify({"error": f"视频过大，请压缩到 {app.config['MAX_STANDARD_MB']}MB 以内"}), 413

    fields = {
        "name": name,
        "japanese_name": (request.form.get("japanese_name") or "").strip(),
        "category": (request.form.get("category") or "").strip(),
        "bilibili": (request.form.get("bilibili") or "").strip(),
        "description": (request.form.get("description") or "").strip(),
    }
    task_id = uuid.uuid4().hex[:10]
    STD_TASKS[task_id] = {
        "task_id": task_id, "status": "queued", "progress": 2,
        "move_name": name, "created_at": datetime.now().isoformat(),
    }
    threading.Thread(
        target=process_standardize,
        args=(task_id, path, fields, file.filename), daemon=True,
    ).start()
    return jsonify({"task_id": task_id, "status": "queued"})


@app.route("/api/admin/upload/<task_id>")
def api_admin_upload_status(task_id: str):
    if not is_admin():
        return jsonify({"error": "需要管理员权限"}), 403
    t = STD_TASKS.get(task_id)
    if not t:
        return jsonify({"error": "任务不存在"}), 404
    resp = {"task_id": t["task_id"], "status": t["status"],
            "progress": t.get("progress", 0), "move_name": t.get("move_name", "")}
    if t["status"] == "done":
        resp["result"] = t["result"]
    elif t["status"] == "error":
        resp["error"] = t.get("error", "未知错误")
    return jsonify(resp)


# ===================================================================
# 健康检查 / 系统信息
# ===================================================================
@app.route("/api/health")
def api_health():
    _, _, _, has_mp = _lazy_fusion()
    db = get_db()
    return jsonify({
        "status": "ok",
        "service": "wota-recognize",
        "technique_count": db.count(),
        "mediapipe_available": bool(has_mp),
        "opencv_available": _check_dep("cv2"),
    })


# ===================================================================
# 预览图静态服务
# ===================================================================
@app.route("/output/<task_id>/<filename>")
def serve_output(task_id: str, filename: str):
    folder = os.path.join(app.config["OUTPUT_FOLDER"], task_id)
    return send_from_directory(folder, filename)


# ===================================================================
# 启动
# ===================================================================
if __name__ == "__main__":
    print("WOTA 动作识别网站启动中（延迟加载模式）...")
    threading.Thread(target=_preload_deps, daemon=True).start()
    print(f"健康检查: http://127.0.0.1:5000/api/health")
    print(f"访问首页: http://127.0.0.1:5000\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
