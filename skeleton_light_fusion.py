"""
Wota艺 骨光融合特征提取 + DTW 比对模块 (核心)
================================================
实现需求中的核心技术：

  【骨光融合提取】
    A路 骨骼：MediaPipe Pose（优先）→ 提取手腕等关键关节坐标
    B路 光棒：OpenCV（HSV / 亮度阈值）→ 提取光斑质心
    性能优化：利用骨骼的"手腕"坐标划定 ROI，只在该区域内找最亮光斑，
             排除背景干扰、省去复杂追踪。

  【暗光环境兜底】
    送入模型前做 CLAHE 自适应直方图均衡化，缓解会场过暗导致骨骼丢失。

  【长视频处理与抽帧】
    动态抽帧：每 2 帧取 1 帧 + 像素差检测，静止帧跳过。
    架构：切片 -> 并行推理 -> 结果聚合（投票 / 时序平滑）。

  【相似度比对】
    DTW（动态时间规整）对逐帧特征序列做时序对齐，输出匹配度。

依赖：
    pip install opencv-python-headless numpy scipy
    pip install mediapipe   # 可选：装上即启用骨骼融合；未装则降级为纯光棒模式
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# MediaPipe 可选：未安装时降级为纯光棒模式
try:
    # ---- 服务器无声卡兼容补丁 ----
    # mediapipe 1.x 导入时会初始化 sounddevice(PortAudio)，无音频设备的环境
    # （如云服务器）会抛 PortAudioError 导致整个 mediapipe 导入失败、误降级。
    # 这里预注入一个最小 sounddevice stub（本项目只用 Pose，不碰音频 API），
    # 让 mediapipe.tasks.python.audio 的 `import sounddevice` 静默通过。
    import sys as _sys
    import types as _types
    if "sounddevice" not in _sys.modules:
        _sd_stub = _types.ModuleType("sounddevice")
        class _StubStream:  # 最小占位：仅保证 import 与类定义不报错
            def __init__(self, *a, **kw):
                pass
            def start(self):
                pass
            def stop(self):
                pass
            def close(self):
                pass
        _sd_stub.InputStream = _StubStream
        _sd_stub.OutputStream = _StubStream
        _sd_stub.RawInputStream = _StubStream
        _sd_stub.RawOutputStream = _StubStream
        _sd_stub.query_devices = lambda *a, **kw: []
        _sys.modules["sounddevice"] = _sd_stub
    import mediapipe as mp  # type: ignore
    _HAS_MP = True
    _MP_IMPORT_ERROR = None
except Exception as _e:
    mp = None
    _HAS_MP = False
    _MP_IMPORT_ERROR = _e

# 探测 mediapipe API 代际：
#   - 0.10.x 提供旧 API `mp.solutions.pose.Pose`（模型内置，无需 .task 文件）
#   - 1.x    移除了 solutions，改用新 API `mp.tasks.python.vision.PoseLandmarker`
#             （需要外部模型文件 pose_landmarker_lite.task）
_MP_API = "none"
if _HAS_MP:
    try:
        from mediapipe.solutions import pose as _mp_pose_mod  # type: ignore
        _MP_API = "legacy"
    except Exception:
        try:
            from mediapipe.tasks.python import vision as _mp_vision  # type: ignore
            from mediapipe.tasks.python.core.base_options import BaseOptions as _MPBaseOptions  # type: ignore
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # type: ignore
            from mediapipe import Image as _MPImage, ImageFormat as _MPImageFormat  # type: ignore
            _MP_API = "tasks"
        except Exception:
            _MP_API = "none"

# PoseLandmarker 模型文件路径（仅新 API 需要）：
#   优先环境变量 MEDIAPIPE_POSE_MODEL，否则取本文件同目录下的 pose_landmarker_lite.task
def _default_model_path() -> str:
    import os
    env = os.environ.get("MEDIAPIPE_POSE_MODEL")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pose_landmarker_lite.task")


def _probe_skeleton_usable() -> bool:
    """探测骨骼提取是否真正可用（mediapipe 导入 + SkeletonExtractor 构造成功）。
    结果缓存，避免每次流水线初始化都重复构造 Pose 对象。"""
    if not _HAS_MP:
        return False
    try:
        se = SkeletonExtractor()
        usable = se.available
        se.close()
        return usable
    except Exception:
        return False


# 全局单例：PoseLandmarker（tasks API）加载极慢（数十秒），进程内只加载一次，
# 所有 worker 共享同一实例，detect 调用用锁串行化保证线程安全。
_pose_landmarker_singleton = None
_pose_landmarker_lock = __import__("threading").Lock()
_pose_landmarker_model_path = None  # 记录已加载的模型路径，用于避免重复加载


def _get_pose_landmarker(min_conf: float = 0.4):
    """懒加载全局单例 PoseLandmarker。首次调用加载模型（慢），之后直接复用。"""
    global _pose_landmarker_singleton, _pose_landmarker_model_path
    if _pose_landmarker_singleton is not None:
        return _pose_landmarker_singleton
    with _pose_landmarker_lock:
        # 双重检查，避免并发重复加载
        if _pose_landmarker_singleton is not None:
            return _pose_landmarker_singleton
        import os
        model_path = _default_model_path()
        if not model_path or not os.path.exists(model_path):
            return None
        options = _mp_vision.PoseLandmarkerOptions(
            base_options=_MPBaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        _pose_landmarker_singleton = _mp_vision.PoseLandmarker.create_from_options(options)
        _pose_landmarker_model_path = model_path
        return _pose_landmarker_singleton


# ===================================================================
# 1. 动态抽帧
# ===================================================================
class FrameSampler:
    """
    动态抽帧：每 step 帧取 1 帧；并用帧间像素差跳过静止帧。
    10 秒 ~300 帧的视频经 step=2 + 静态跳过后通常降到 ~80~150 帧。
    """

    def __init__(self, step: int = 2, max_frames: int = 150,
                 static_changed_frac: float = 0.0008,
                 pixel_diff: int = 15, target_fps: float = 30.0):
        self.step = max(1, step)
        self.max_frames = max_frames
        # 静止判定：用“显著变化像素占比”，而非全图均值。
        # wota 场景中光棒只占画面一小部分，全图均值会被大片暗背景稀释，
        # 误判为静止；改用变化像素占比更鲁棒。
        self.static_changed_frac = static_changed_frac
        self.pixel_diff = pixel_diff
        self.target_fps = target_fps

    def sample(self, video_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: List[np.ndarray] = []
        idx = 0
        last_gray: Optional[np.ndarray] = None
        skipped_static = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if last_gray is not None:
                    changed_frac = float(np.mean(
                        cv2.absdiff(gray, last_gray) > self.pixel_diff
                    ))
                    if changed_frac < self.static_changed_frac:
                        # 静止帧：跳过（仍更新 last_gray 以连续判断）
                        last_gray = gray
                        idx += 1
                        skipped_static += 1
                        continue
                frames.append(frame)
                last_gray = gray
                if len(frames) >= self.max_frames:
                    break
            idx += 1
        cap.release()

        # 若抽帧后仍超长，再均匀采样到 max_frames
        if len(frames) > self.max_frames:
            sel = np.linspace(0, len(frames) - 1, self.max_frames).astype(int)
            frames = [frames[i] for i in sel]

        meta = {
            "fps": float(fps),
            "total_frames": total,
            "width": w,
            "height": h,
            "sampled": len(frames),
            "skipped_static": skipped_static,
            "duration": (total / fps) if fps > 0 else 0.0,
        }
        return frames, meta


# ===================================================================
# 2. 暗光增强 (CLAHE)
# ===================================================================
class CLAHEEnhancer:
    """在 L 通道做 CLAHE，增强暗光会场的局部对比度。"""

    def __init__(self, clip_limit: float = 3.0, tile_grid: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ===================================================================
# 3. 骨骼提取 (MediaPipe, 可选)
# ===================================================================
# MediaPipe Pose 关键点索引
LM_NOSE = 0
LM_L_SHOULDER, LM_R_SHOULDER = 11, 12
LM_L_ELBOW, LM_R_ELBOW = 13, 14
LM_L_WRIST, LM_R_WRIST = 15, 16
LM_L_HIP, LM_R_HIP = 23, 24


@dataclass
class Skeleton:
    """单帧骨骼关键点（归一化坐标 0~1，基于图像宽高）。None 表示未检测到。"""
    found: bool = False
    l_wrist: Optional[Tuple[float, float]] = None
    r_wrist: Optional[Tuple[float, float]] = None
    l_elbow: Optional[Tuple[float, float]] = None
    r_elbow: Optional[Tuple[float, float]] = None
    l_shoulder: Optional[Tuple[float, float]] = None
    r_shoulder: Optional[Tuple[float, float]] = None
    l_hip: Optional[Tuple[float, float]] = None
    r_hip: Optional[Tuple[float, float]] = None
    l_wrist_vis: float = 0.0
    r_wrist_vis: float = 0.0

    @staticmethod
    def _xy(lm):
        return (float(lm.x), float(lm.y))


class SkeletonExtractor:
    """
    MediaPipe Pose 封装。每个实例拥有自己的 Pose 对象，
    便于在切片并行中每线程独立使用（避免线程安全问题）。
    兼容 mediapipe 两代 API：
      - legacy (0.10.x): mp.solutions.pose.Pose（模型内置）
      - tasks  (1.x):    PoseLandmarker（需要 pose_landmarker_lite.task 模型文件）
    未安装 mediapipe 或模型文件缺失时 available=False，extract 返回空 Skeleton。
    """

    def __init__(self, model_complexity: int = 0, min_conf: float = 0.4):
        self.available = False
        self.pose = None
        self._api = "none"
        self._owns_pose = False  # 是否本实例独占 pose（legacy 为 True；tasks 共享单例为 False）
        if not _HAS_MP:
            return
        try:
            if _MP_API == "legacy":
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=min_conf,
                )
                self._api = "legacy"
                self._owns_pose = True
                self.available = True
            elif _MP_API == "tasks":
                self.pose = _get_pose_landmarker(min_conf)
                if self.pose is None:
                    self.available = False
                    return
                self._api = "tasks"
                self._owns_pose = False
                self.available = True
        except Exception:
            self.available = False
            self.pose = None
            self._api = "none"

    def extract(self, frame_bgr: np.ndarray) -> Skeleton:
        sk = Skeleton()
        if not self.available or self.pose is None:
            return sk
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self._api == "legacy":
                res = self.pose.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    self._fill(sk, lm)
            elif self._api == "tasks":
                mp_img = _MPImage(image_format=_MPImageFormat.SRGB, data=rgb)
                # 单例共享：detect 串行化，保证线程安全
                with _pose_landmarker_lock:
                    res = self.pose.detect(mp_img)
                if res.pose_landmarks and len(res.pose_landmarks) > 0:
                    lm = res.pose_landmarks[0]
                    self._fill(sk, lm)
        except Exception:
            pass
        return sk

    def _fill(self, sk: Skeleton, lm) -> None:
        """从 landmark 列表填充 Skeleton（两代 API 的 landmark 均含 .x/.y/.visibility）。"""
        sk.found = True
        sk.l_shoulder = Skeleton._xy(lm[LM_L_SHOULDER])
        sk.r_shoulder = Skeleton._xy(lm[LM_R_SHOULDER])
        sk.l_elbow = Skeleton._xy(lm[LM_L_ELBOW])
        sk.r_elbow = Skeleton._xy(lm[LM_R_ELBOW])
        sk.l_hip = Skeleton._xy(lm[LM_L_HIP])
        sk.r_hip = Skeleton._xy(lm[LM_R_HIP])
        sk.l_wrist = Skeleton._xy(lm[LM_L_WRIST])
        sk.r_wrist = Skeleton._xy(lm[LM_R_WRIST])
        sk.l_wrist_vis = float(lm[LM_L_WRIST].visibility)
        sk.r_wrist_vis = float(lm[LM_R_WRIST].visibility)

    def close(self):
        # tasks 单例共享：不关闭（其他实例还在用）；仅 legacy 独占实例才关闭
        if self.pose is not None and self._owns_pose:
            try:
                self.pose.close()
            except Exception:
                pass


# ===================================================================
# 4. 光斑检测 (OpenCV, ROI 优化)
# ===================================================================
@dataclass
class LightSpot:
    found: bool = False
    x: float = 0.0          # 归一化 x (0~1)
    y: float = 0.0          # 归一化 y (0~1)
    brightness: float = 0.0 # 0~1
    area_norm: float = 0.0  # 面积 / 帧面积
    orientation: float = 0.0  # 光斑主轴方向角(弧度 0~π)，由轮廓椭圆拟合得到
    elongation: float = 1.0   # 椭圆长轴/短轴比 (>=1)，1=圆形，越大越接近光弧拖影


class LightDetector:
    """
    HSV / 亮度阈值提取光斑。核心优化：只在手腕 ROI 内寻找最亮光斑。
    无手腕时回退到画面上 2/3 区域全局搜索。
    """

    def __init__(self, v_threshold: int = 200, s_max: int = 90,
                 roi_radius: float = 0.14, min_area: int = 6):
        self.v_threshold = v_threshold
        self.s_max = s_max
        self.roi_radius = roi_radius   # ROI 半径（归一化）
        self.min_area = min_area

    def _detect_in_roi(self, frame_bgr: np.ndarray,
                       roi: Tuple[int, int, int, int]) -> Optional[LightSpot]:
        x0, y0, x1, y1 = roi
        h, w = frame_bgr.shape[:2]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w, x1); y1 = min(h, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        patch = frame_bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        # 高亮 & 低~中饱和（化学光棒常见特征）；白色光棒饱和低
        mask = (v >= self.v_threshold) & (s <= self.s_max)
        mask = mask.astype(np.uint8) * 255
        if int(mask.sum()) == 0:
            # 退而求其次：取该 ROI 内最亮像素
            idx = np.unravel_index(np.argmax(v), v.shape)
            if v[idx] < self.v_threshold - 40:
                return None
            cx = (x0 + idx[1]) / w
            cy = (y0 + idx[0]) / h
            return LightSpot(True, float(cx), float(cy),
                             float(v[idx]) / 255.0, 0.0, 0.0, 1.0)
        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # 选最大且最亮的连通域
        best = None
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx_p = M["m10"] / M["m00"]
            cy_p = M["m01"] / M["m00"]
            mean_v = float(v[
                np.clip(int(cy_p), 0, v.shape[0] - 1),
                np.clip(int(cx_p), 0, v.shape[1] - 1),
            ]) / 255.0
            cand = (area, mean_v, cx_p, cy_p)
            if best is None or (cand[1], cand[0]) > (best[1], best[0]):
                best = cand
                best_cnt = cnt
        if best is None:
            return None
        area, mean_v, cx_p, cy_p = best
        # ---- 光弧几何特征：椭圆拟合得到主轴方向 + 椭圆度 ----
        # 光棒拖影通常呈拉长条状，其主轴方向反映"横挥/竖挥/斜挥"，
        # 椭圆度(长轴/短轴)反映"点状光斑 vs 拉长的光弧"。
        orientation = 0.0
        elongation = 1.0
        if best_cnt is not None and len(best_cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(best_cnt)
                (_, _), (major, minor), angle = ellipse
                if minor > 1e-6:
                    elongation = float(major) / float(minor)
                    elongation = min(elongation, 20.0)  # 裁剪异常值
                # angle 是度(0~180)，转弧度(0~π)
                orientation = float(np.deg2rad(angle))
            except Exception:
                orientation = 0.0
                elongation = 1.0
        return LightSpot(
            True,
            float((x0 + cx_p) / w),
            float((y0 + cy_p) / h),
            float(mean_v),
            float(area) / float(w * h),
            orientation,
            elongation,
        )

    def detect_global(self, frame_bgr: np.ndarray,
                      upper_ratio: float = 2.0 / 3.0) -> Optional[LightSpot]:
        """全局（默认画面上 2/3）搜索最亮光斑，作为无手腕时的兜底。"""
        h, w = frame_bgr.shape[:2]
        y_end = max(1, int(h * upper_ratio))
        sub = frame_bgr[:y_end, :]
        return self._detect_in_roi(sub, (0, 0, w, y_end))

    def detect_near_wrist(self, frame_bgr: np.ndarray,
                          wrist: Optional[Tuple[float, float]],
                          visibility: float = 0.0) -> Optional[LightSpot]:
        """
        核心优化：在手腕周围 ROI 内找最亮光斑，排除背景干扰。
        wrist 为归一化坐标；visibility 不足时扩大搜索范围；无手腕则全局兜底。
        """
        h, w = frame_bgr.shape[:2]
        if wrist is None or visibility < 0.2:
            return self.detect_global(frame_bgr)
        wx, wy = wrist
        r = self.roi_radius * (1.5 if visibility < 0.4 else 1.0)
        x0 = int((wx - r) * w); y0 = int((wy - r) * h)
        x1 = int((wx + r) * w); y1 = int((wy + r) * h)
        spot = self._detect_in_roi(frame_bgr, (x0, y0, x1, y1))
        if spot is None:
            # ROI 内没找到 -> 全局兜底
            spot = self.detect_global(frame_bgr)
        return spot


def _detect_near_wrist(detector: "LightDetector", frame_bgr: np.ndarray,
                       wrist: Optional[Tuple[float, float]],
                       visibility: float = 0.0) -> Optional[LightSpot]:
    """模块级薄包装，调用 LightDetector.detect_near_wrist。"""
    return detector.detect_near_wrist(frame_bgr, wrist, visibility)


# ===================================================================
# 5. 逐帧特征构建 (固定 28 维)
# ===================================================================
class FeatureBuilder:
    """
    将一帧的骨骼 + 光斑融合为固定 28 维特征向量：
      骨骼(18): 左/右手腕、肘、肩 (x,y)×6 + 左/右臂角 + 左/右臂展
               + 双腕距离 + 身体倾斜
      光棒(10): 左/右光斑相对手腕偏移(dx,dy)、亮度、面积 + 左/右光斑速度
    无骨骼时骨骼维置 0，光斑用全局坐标（部署内一致即可比对）。
    """

    DIM = 38

    @staticmethod
    def _angle(p, c, q) -> float:
        """点 c 处由 p-c-q 形成的夹角(度)。任一缺失返回 180。"""
        if p is None or c is None or q is None:
            return 180.0
        v1 = np.array(p) - np.array(c)
        v2 = np.array(q) - np.array(c)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
        return float(np.degrees(np.arccos(cos)))

    @staticmethod
    def _dist(a, b) -> float:
        if a is None or b is None:
            return 0.0
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def build(self, sk: Skeleton, l_light: Optional[LightSpot],
              r_light: Optional[LightSpot],
              prev_l_light: Optional[LightSpot],
              prev_r_light: Optional[LightSpot]) -> List[float]:
        f = [0.0] * self.DIM

        # ---- 骨骼 ----
        if sk.found:
            lw, rw = sk.l_wrist, sk.r_wrist
            le, re = sk.l_elbow, sk.r_elbow
            ls, rs = sk.l_shoulder, sk.r_shoulder
            lh, rh = sk.l_hip, sk.r_hip
            if lw: f[0], f[1] = lw
            if rw: f[2], f[3] = rw
            if le: f[4], f[5] = le
            if re: f[6], f[7] = re
            if ls: f[8], f[9] = ls
            if rs: f[10], f[11] = rs
            # 臂角 (肘处 shoulder-elbow-wrist)
            f[12] = self._angle(ls, le, lw) / 180.0
            f[13] = self._angle(rs, re, rw) / 180.0
            # 臂展 = 肩->腕距离 / 躯干高
            torso_h = self._dist(
                ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2) if ls and rs else None,
                ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2) if lh and rh else None,
            )
            if torso_h > 1e-6:
                f[14] = self._dist(ls, lw) / torso_h
                f[15] = self._dist(rs, rw) / torso_h
            # 双腕距离 (归一化到 ~0~1)
            f[16] = min(self._dist(lw, rw) / 1.5, 1.0) if lw and rw else 0.0
            # 身体倾斜 (肩连线角度 / pi)
            if ls and rs:
                f[17] = float(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]) / np.pi)

        # ---- 光棒 (每侧 10 维) ----
        # slot 布局: [0]=dx [1]=dy [2]=brightness [3]=area [4]=speed
        #            [5]=orient_sin [6]=orient_cos [7]=elongation_norm
        #            [8]=vel_dir_sin [9]=vel_dir_cos
        def fill_light(slot, light, wrist, prev_light):
            if light is not None and light.found:
                if wrist is not None:
                    f[slot] = light.x - wrist[0]
                    f[slot + 1] = light.y - wrist[1]
                else:
                    # 无手腕：存全局归一化坐标
                    f[slot] = light.x
                    f[slot + 1] = light.y
                f[slot + 2] = light.brightness
                f[slot + 3] = light.area_norm
                if prev_light is not None and prev_light.found:
                    f[slot + 4] = min(
                        np.hypot(light.x - prev_light.x, light.y - prev_light.y) / 0.5,
                        1.0,
                    )
                # 光弧主轴方向：用 sin(2θ)/cos(2θ) 消除 0/π 的符号跳变
                f[slot + 5] = float(np.sin(2.0 * light.orientation))
                f[slot + 6] = float(np.cos(2.0 * light.orientation))
                # 椭圆度归一化：1(点状) -> 0，越大越接近拉长光弧 -> 1
                f[slot + 7] = min((light.elongation - 1.0) / 9.0, 1.0)
                # 光斑速度方向：刻画光弧运动走向/曲率
                if prev_light is not None and prev_light.found:
                    vx = light.x - prev_light.x
                    vy = light.y - prev_light.y
                    mag = np.hypot(vx, vy)
                    if mag > 1e-6:
                        ang = np.arctan2(vy, vx)
                        f[slot + 8] = float(np.sin(ang))
                        f[slot + 9] = float(np.cos(ang))

        fill_light(18, l_light, sk.l_wrist if sk.found else None, prev_l_light)
        fill_light(28, r_light, sk.r_wrist if sk.found else None, prev_r_light)
        return f


# ===================================================================
# 6. 切片并行 worker
# ===================================================================
class SliceWorker:
    """处理一个帧切片：增强 -> 骨骼 -> 光斑 -> 特征。每 worker 独占 Pose。"""

    def __init__(self, enhancer: CLAHEEnhancer, use_skeleton: bool,
                 model_complexity: int = 0):
        self.enhancer = enhancer
        if use_skeleton:
            self.skeleton = SkeletonExtractor(model_complexity=model_complexity)
        else:
            self.skeleton = None  # 纯光棒模式：完全不构造 Pose（避免加载模型）
        self.light = LightDetector()
        self.builder = FeatureBuilder()

    def process(self, frames: List[np.ndarray]) -> List[List[float]]:
        seq: List[List[float]] = []
        prev_l: Optional[LightSpot] = None
        prev_r: Optional[LightSpot] = None
        last_sk: Optional[Skeleton] = None  # 用于缺失帧的时序平滑（继承上一帧）
        carry = 0
        for frame in frames:
            enhanced = self.enhancer.enhance(frame)
            sk = self.skeleton.extract(enhanced) if self.skeleton is not None else Skeleton()
            if not sk.found and last_sk is not None and carry < 5:
                # 短暂丢失：沿用上一帧骨骼（时序平滑兜底）
                sk = last_sk
                carry += 1
            else:
                last_sk = sk if sk.found else last_sk
                carry = 0 if sk.found else carry

            l_light = _detect_near_wrist(self.light, enhanced, sk.l_wrist, sk.l_wrist_vis)
            r_light = _detect_near_wrist(self.light, enhanced, sk.r_wrist, sk.r_wrist_vis)

            vec = self.builder.build(sk, l_light, r_light, prev_l, prev_r)
            seq.append(vec)
            prev_l, prev_r = l_light, r_light
        return seq

    def close(self):
        if self.skeleton is not None:
            self.skeleton.close()


# ===================================================================
# 7. 主流水线：切片 -> 并行推理 -> 结果聚合
# ===================================================================
class FusionPipeline:
    """一键骨光融合提取，输出逐帧特征序列。"""

    def __init__(self, step: int = 2, max_frames: int = 150,
                 n_workers: int = 4, use_skeleton: bool = True,
                 model_complexity: int = 0):
        self.sampler = FrameSampler(step=step, max_frames=max_frames)
        self.enhancer = CLAHEEnhancer()
        self.n_workers = max(1, n_workers)
        # 真实骨骼可用性：mediapipe 已导入 且 SkeletonExtractor 能成功构造
        # （新 API 还需模型文件存在；旧 API 无需）。仅在使用骨骼时探测一次。
        if use_skeleton:
            self._skeleton_usable = _probe_skeleton_usable()
        else:
            self._skeleton_usable = False
        self.use_skeleton = use_skeleton and self._skeleton_usable
        self.model_complexity = model_complexity
        self.meta: Dict[str, Any] = {}

    @property
    def has_mediapipe(self) -> bool:
        return self._skeleton_usable

    def _split_slices(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        n = len(frames)
        if n == 0:
            return []
        k = min(self.n_workers, n)
        bounds = np.linspace(0, n, k + 1).astype(int)
        return [frames[bounds[i]:bounds[i + 1]] for i in range(k)]

    def _smooth(self, seq: List[List[float]]) -> List[List[float]]:
        """时序平滑：滑动均值（结果聚合的一部分）。"""
        if len(seq) < 5:
            return seq
        arr = np.array(seq, dtype=np.float32)
        out = arr.copy()
        w = 3
        for i in range(arr.shape[1]):
            out[:, i] = np.convolve(arr[:, i], np.ones(w) / w, mode="same")
        # 用 savgol 做更平滑的处理（若可用）
        if HAS_SCIPY and len(arr) >= 7:
            win = min(7, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
            if win >= 5:
                for i in range(arr.shape[1]):
                    out[:, i] = savgol_filter(arr[:, i], win, 2)
        return out.tolist()

    def extract(self, video_path: str) -> Dict[str, Any]:
        frames, meta = self.sampler.sample(video_path)
        self.meta = meta
        if len(frames) < 2:
            raise ValueError("抽帧后帧数不足（至少 2 帧），请上传更长的视频")

        slices = self._split_slices(frames)
        # 并行推理
        workers: List[SliceWorker] = []
        results: List[Optional[List[List[float]]]] = [None] * len(slices)
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futs = {}
            for i, sl in enumerate(slices):
                w = SliceWorker(self.enhancer, self.use_skeleton, self.model_complexity)
                workers.append(w)
                futs[ex.submit(w.process, sl)] = i
            for fut in futs:
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"[Fusion] 切片 {idx} 处理失败: {e}")
                    results[idx] = []
        for w in workers:
            w.close()

        # 结果聚合：按时序拼接
        seq: List[List[float]] = []
        for r in results:
            if r:
                seq.extend(r)

        seq = self._smooth(seq)
        meta["feature_frames"] = len(seq)
        meta["use_skeleton"] = self.use_skeleton
        return {
            "feature_sequence": seq,
            "frame_count": len(seq),
            "duration": meta.get("duration", 0.0),
            "fps": meta.get("fps", 30.0),
            "meta": meta,
        }

    def render_preview(self, video_path: str, out_path: str) -> Optional[str]:
        """在中间帧上叠加骨骼+光斑，生成预览图。"""
        try:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            enh = CLAHEEnhancer().enhance(frame)
            sk = SkeletonExtractor().extract(enh)
            ld = LightDetector()
            canvas = frame.copy()
            if sk.found:
                for p in [sk.l_shoulder, sk.r_shoulder, sk.l_elbow, sk.r_elbow,
                          sk.l_wrist, sk.r_wrist, sk.l_hip, sk.r_hip]:
                    if p:
                        cv2.circle(canvas, (int(p[0] * canvas.shape[1]),
                                            int(p[1] * canvas.shape[0])),
                                   4, (0, 255, 255), -1)
                # 连线：肩-肘-腕
                for a, b in [(sk.l_shoulder, sk.l_elbow), (sk.l_elbow, sk.l_wrist),
                             (sk.r_shoulder, sk.r_elbow), (sk.r_elbow, sk.r_wrist)]:
                    if a and b:
                        cv2.line(canvas,
                                 (int(a[0] * canvas.shape[1]), int(a[1] * canvas.shape[0])),
                                 (int(b[0] * canvas.shape[1]), int(b[1] * canvas.shape[0])),
                                 (0, 255, 255), 2)
            for light, col in [(_detect_near_wrist(ld, enh, sk.l_wrist, sk.l_wrist_vis), (0, 0, 255)),
                               (_detect_near_wrist(ld, enh, sk.r_wrist, sk.r_wrist_vis), (255, 0, 255))]:
                if light and light.found:
                    cv2.circle(canvas, (int(light.x * canvas.shape[1]),
                                        int(light.y * canvas.shape[0])),
                               9, col, 2)
                    cv2.circle(canvas, (int(light.x * canvas.shape[1]),
                                        int(light.y * canvas.shape[0])),
                               3, col, -1)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            cv2.imwrite(out_path, canvas)
            return out_path
        except Exception as e:
            print(f"[Fusion] 预览图生成失败: {e}")
            return None


# ===================================================================
# 8. DTW 时序比对
# ===================================================================
class DTWMatcher:
    """
    带状约束 (Sakoe-Chiba) 的 DTW。返回平均步代价与路径长度。
    相似度 = exp(-avg_dist / sigma)，匹配度百分比 = 相似度 * 100。
    """

    # 默认权重：光棒维稍高（wota 的光棒轨迹是核心签名）
    # 38 维 = 骨骼 18 + 光棒 20
    # 光棒每侧 10 维布局: dx,dy,brightness,area,speed,orient_sin,orient_cos,elongation,vel_dir_sin,vel_dir_cos
    # 光弧形状维(orient/elongation/vel_dir)给更高权重以强化形状判别
    _LIGHT_SIDE_WEIGHTS = [1.3, 1.3, 1.0, 1.0, 1.3, 1.5, 1.5, 1.4, 1.5, 1.5]
    DEFAULT_WEIGHTS = np.array(
        [1.0] * 18 + _LIGHT_SIDE_WEIGHTS * 2, dtype=np.float32
    )

    def __init__(self, sigma: float = 0.35, band: float = 0.3,
                 weights: Optional[np.ndarray] = None):
        self.sigma = sigma
        self.band = band
        self.weights = (weights if weights is not None else self.DEFAULT_WEIGHTS)

    def _cost_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        w = self.weights
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt(np.sum((w * diff) ** 2, axis=2)).astype(np.float64)

    def align(self, seq_a: List[List[float]], seq_b: List[List[float]]
               ) -> Tuple[float, int]:
        A = np.asarray(seq_a, dtype=np.float32)
        B = np.asarray(seq_b, dtype=np.float32)
        if A.size == 0 or B.size == 0:
            return float("inf"), 0
        # 维度不匹配（如旧版 28 维数据 vs 新版 38 维）时直接判为不可比，
        # 避免 numpy 广播报错导致整个识别崩溃。
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
            return float("inf"), 0
        n, m = A.shape[0], B.shape[0]
        cost = self._cost_matrix(A, B)
        INF = float("inf")
        D = np.full((n + 1, m + 1), INF, dtype=np.float64)
        D[0, 0] = 0.0
        wband = max(1, int(self.band * max(n, m)))
        for i in range(1, n + 1):
            j_start = max(1, i - wband)
            j_end = min(m, i + wband)
            for j in range(j_start, j_end + 1):
                D[i, j] = cost[i - 1, j - 1] + min(
                    D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]
                )
        # 回溯路径长度
        i, j = n, m
        path_len = 0
        total = 0.0
        while i > 0 and j > 0:
            total += cost[i - 1, j - 1]
            path_len += 1
            if i == 1 and j == 1:
                break
            cands = []
            if i > 1:
                cands.append((D[i - 1, j], i - 1, j))
            if j > 1:
                cands.append((D[i, j - 1], i, j - 1))
            if i > 1 and j > 1:
                cands.append((D[i - 1, j - 1], i - 1, j - 1))
            _, i, j = min(cands)
        avg = total / max(path_len, 1)
        return float(avg), path_len

    def similarity(self, seq_a: List[List[float]], seq_b: List[List[float]]) -> float:
        avg, _ = self.align(seq_a, seq_b)
        if avg == float("inf"):
            return 0.0
        return float(np.exp(-avg / self.sigma))


# ===================================================================
# 9. 识别器：切片投票 + 全序列 DTW 聚合
# ===================================================================
class Recognizer:
    """
    给定查询特征序列与技术库，返回 Top-K 匹配。
    聚合策略：
      - 全序列 DTW 相似度 (主)
      - 切片投票 (投票/时序平滑)：查询切 K 段，每段独立 DTW 比对，
        票投给该段最佳技术；vote_fraction = 票数 / K
      - 综合 = 0.7 * full_sim + 0.3 * vote_fraction
    """

    def __init__(self, matcher: Optional[DTWMatcher] = None, n_slices: int = 3,
                 min_score: float = 0.25, top_k: int = 5):
        self.matcher = matcher or DTWMatcher()
        self.n_slices = max(1, n_slices)
        self.min_score = min_score
        self.top_k = top_k

    def _slice_seq(self, seq: List[List[float]]) -> List[List[List[float]]]:
        n = len(seq)
        if n == 0:
            return []
        k = min(self.n_slices, n)
        bounds = np.linspace(0, n, k + 1).astype(int)
        return [seq[bounds[i]:bounds[i + 1]] for i in range(k) if bounds[i] < bounds[i + 1]]

    def recognize(self, query_seq: List[List[float]],
                  candidates: List[Tuple[Dict, List[List[float]]]]
                  ) -> List[Dict]:
        """
        candidates: [(meta_dict, feature_sequence), ...]
        返回 [{"move_id","move_name","japanese_name","category","bilibili",
               "match","full_sim","vote_fraction"}, ...] 按 match 降序。
        """
        if not query_seq or not candidates:
            return []

        q_slices = self._slice_seq(query_seq)
        results: List[Dict] = []
        for meta, cseq in candidates:
            if not cseq:
                continue
            full_sim = self.matcher.similarity(query_seq, cseq)
            # 切片投票
            votes = 0
            for qs in q_slices:
                best_sim, best_id = -1.0, None
                for m2, s2 in candidates:
                    if not s2:
                        continue
                    s = self.matcher.similarity(qs, s2)
                    if s > best_sim:
                        best_sim, best_id = s, m2["move_id"]
                if best_id == meta["move_id"]:
                    votes += 1
            vote_frac = votes / max(len(q_slices), 1)
            combined = 0.7 * full_sim + 0.3 * vote_frac
            results.append({
                "move_id": meta["move_id"],
                "move_name": meta["move_name"],
                "japanese_name": meta.get("japanese_name", ""),
                "category": meta.get("category", ""),
                "bilibili": meta.get("bilibili", ""),
                "description": meta.get("description", ""),
                "match": round(combined * 100, 1),
                "full_sim": round(full_sim * 100, 1),
                "vote_fraction": round(vote_frac * 100, 1),
            })
        results.sort(key=lambda r: r["match"], reverse=True)
        # 过滤最低阈值
        results = [r for r in results if r["match"] >= self.min_score * 100]
        return results[: self.top_k]


# ===================================================================
# 10. 命令行：提取 / 比对
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wota艺 骨光融合特征提取")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("-o", "--output", default="./output", help="输出目录")
    parser.add_argument("--step", type=int, default=2, help="抽帧步长")
    parser.add_argument("-n", "--max-frames", type=int, default=150, help="最大帧数")
    parser.add_argument("--workers", type=int, default=4, help="并行切片数")
    parser.add_argument("--no-skeleton", action="store_true", help="禁用骨骼(纯光棒模式)")
    args = parser.parse_args()

    pipe = FusionPipeline(step=args.step, max_frames=args.max_frames,
                          n_workers=args.workers,
                          use_skeleton=not args.no_skeleton)
    print(f"MediaPipe 可用: {pipe.has_mediapipe}  使用骨骼: {pipe.use_skeleton}")
    res = pipe.extract(args.video)
    os.makedirs(args.output, exist_ok=True)
    prev = pipe.render_preview(args.video, os.path.join(args.output, "fusion_preview.png"))
    print(f"抽取 {res['frame_count']} 帧特征 (38维/帧), 时长 {res['duration']:.1f}s")
    print(f"预览图: {prev}")
    seq = res["feature_sequence"]
    print(f"首帧特征: {[round(v,3) for v in seq[0]]}")
