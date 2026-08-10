"""
Wota艺 光流提取模块
=====================
核心功能：
  1. 视频抽帧
  2. ROI 掩膜 —— 基于 HSV 色彩阈值提取光棒（发光区域）
  3. 光流计算 —— RAFT（优先） / Farneback（回退）
  4. 轨迹追踪 —— 对光流关键点进行逐帧追踪
  5. 轨迹特征 —— 曲率、闭合度、运动方向、速度
  6. 时序平滑 —— 卡尔曼滤波 / 滑动窗口
  7. 可视化输出

依赖（详见末尾 requirements）：
  torch, torchvision, opencv-python, numpy, scipy, matplotlib, tqdm
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import warnings

# ---------------------------------------------------------------------------
# 0. 可选依赖导入
# ---------------------------------------------------------------------------
try:
    import torch
    import torchvision.transforms.functional as F
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    HAS_RAFT = True
except ImportError:
    HAS_RAFT = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *a, **kw):
        return iterable


# ===================================================================
# 1. 视频抽帧
# ===================================================================
class VideoReader:
    """读取视频，按步长抽帧 或 按目标帧数均匀抽帧。"""

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract(self, step: int = 1, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        抽帧。
        - step: 每隔 step 帧取一帧
        - max_frames: 最多取多少帧（均匀采样）
        """
        if max_frames is not None and max_frames < self.total_frames:
            step = max(1, self.total_frames // max_frames)

        frames = []
        idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
        self.cap.release()
        return frames

    def __repr__(self):
        return f"VideoReader(fps={self.fps:.1f}, frames={self.total_frames}, {self.width}x{self.height})"


# ===================================================================
# 2. ROI 掩膜 —— 提取发光区域（光棒）
# ===================================================================
class ROIMasker:
    """
    基于 HSV / 亮度 阈值提取画面中的发光区域（光棒轨迹）。
    Wota艺的化学光棒通常呈高饱和度、高明度颜色（青、橙、粉、绿等）。
    """

    # 预设常见光棒颜色范围（HSV）
    PRESETS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "orange": (np.array([11, 150, 150]),  np.array([25, 255, 255])),
        "blue":   (np.array([100, 120, 150]), np.array([130, 255, 255])),
        "red":    (np.array([0, 150, 150]),   np.array([10, 255, 255])),
        "pink":   (np.array([150, 100, 150]), np.array([175, 255, 255])),
        "green":  (np.array([40, 120, 120]),  np.array([80, 255, 255])),
        "white":  (np.array([0, 0, 200]),     np.array([180, 40, 255])),   # 低饱和高明度
        "purple": (np.array([135, 100, 120]), np.array([160, 255, 255])),
    }

    def __init__(self, color_presets: List[str] = None, custom_lower=None, custom_upper=None):
        """
        color_presets: 预设名称列表，如 ["orange", "blue", "red"]
        """
        self.masks = []
        if custom_lower is not None and custom_upper is not None:
            self.masks.append((np.array(custom_lower), np.array(custom_upper)))
        if color_presets:
            for name in color_presets:
                if name in self.PRESETS:
                    self.masks.append(self.PRESETS[name])

        if not self.masks:
            # 默认：通用高明度 + 高饱和
            self.masks.append(self.PRESETS["white"])

    def apply(self, frame: np.ndarray, morph: bool = True) -> np.ndarray:
        """
        返回 0/1 掩膜 (H, W)，1 表示光棒区域。
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined = np.zeros(frame.shape[:2], dtype=np.uint8)

        for lower, upper in self.masks:
            mask = cv2.inRange(hsv, lower, upper)
            combined = cv2.bitwise_or(combined, mask)

        if morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

        return combined

    @staticmethod
    def from_sample(frame: np.ndarray) -> "ROIMasker":
        """
        简单自适应：取画面中亮度前 5% 的像素作为阈值参考。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, 95)
        lower = np.array([0, 0, threshold], dtype=np.uint8)
        upper = np.array([180, 80, 255], dtype=np.uint8)
        return ROIMasker(custom_lower=lower, custom_upper=upper)


# ===================================================================
# 3. 光流计算引擎
# ===================================================================
class OpticalFlowEngine:
    """
    光流计算。
    - 优先使用 RAFT（torchvision）
    - 回退使用 Farneback（OpenCV）
    """

    def __init__(self, method: str = "raft", device: str = "cuda"):
        self.method = method.lower()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        if self.method == "raft":
            if not HAS_RAFT:
                warnings.warn("RAFT 不可用，回退到 Farneback")
                self.method = "farneback"
            else:
                self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
                self.model.to(self.device)
                self.model.eval()

    def compute(self, prev_frame: np.ndarray, curr_frame: np.ndarray, mask: np.ndarray = None):
        """
        计算两张连续帧之间的光流。
        返回 (flow, flow_vis)：
          - flow: (H, W, 2)  float32 numpy
          - flow_vis: (H, W, 3) uint8 HSV可视化
        """
        if self.method == "raft":
            flow = self._raft_flow(prev_frame, curr_frame)
        else:
            flow = self._farneback_flow(prev_frame, curr_frame)

        # ROI 掩膜过滤：非光棒区域置零
        if mask is not None:
            flow[~mask.astype(bool)] = 0.0

        flow_vis = self._flow_to_vis(flow)
        return flow, flow_vis

    def _raft_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """RAFT 光流"""
        img1 = torch.from_numpy(prev).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        img2 = torch.from_numpy(curr).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            flow = self.model(img1, img2)[-1]  # (1, 2, H, W)
        flow = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return flow.astype(np.float32)

    def _farneback_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Gunnar Farneback 稠密光流"""
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow  # (H, W, 2)

    @staticmethod
    def _flow_to_vis(flow: np.ndarray) -> np.ndarray:
        """光流 -> HSV 色轮可视化"""
        h, w = flow.shape[:2]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ===================================================================
# 4. 轨迹追踪器
# ===================================================================
@dataclass
class Trajectory:
    """单条轨迹"""
    points: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    directions: List[float] = field(default_factory=list)  # 弧度

    def add(self, x: float, y: float, frame_idx: int,
            vx: float = 0, vy: float = 0):
        self.points.append((x, y))
        self.timestamps.append(frame_idx)
        self.velocities.append(np.sqrt(vx ** 2 + vy ** 2))
        self.directions.append(np.arctan2(vy, vx))

    def __len__(self):
        return len(self.points)

    @property
    def last_point(self) -> Optional[Tuple[float, float]]:
        return self.points[-1] if self.points else None


class TrajectoryTracker:
    """
    基于光流追踪发光区域（光斑）质心的运动轨迹。
    策略：对 ROI 掩膜中的每个连通区域计算质心，逐帧关联最近质心形成轨迹。
    """

    def __init__(self, max_disappear: int = 5, min_area: int = 50):
        self.max_disappear = max_disappear
        self.min_area = min_area
        self.trajectories: List[Trajectory] = []
        self.active: Dict[int, Trajectory] = {}  # track_id -> Trajectory
        self.next_id = 0

    def update(self, mask: np.ndarray, flow: np.ndarray, frame_idx: int):
        """
        每帧调用：从 ROI mask 中提取光斑质心，关联已有轨迹。
        """
        # 找连通区域质心
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroids.append((cx, cy))

        if not centroids:
            # 全部标记消失
            for tid in list(self.active.keys()):
                self.active.pop(tid)
            return

        # 关联：最近邻
        used = set()
        for cx, cy in centroids:
            best_id = None
            best_dist = float("inf")
            for tid, traj in self.active.items():
                if tid in used:
                    continue
                if traj.last_point is None:
                    continue
                lx, ly = traj.last_point
                dist = np.hypot(cx - lx, cy - ly)
                # 光流一致性约束：偏差过大则认为不是同一轨迹
                if dist < 80 and dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                traj = self.active[best_id]
                traj.add(cx, cy, frame_idx)
                used.add(best_id)
            else:
                # 新轨迹
                new_traj = Trajectory()
                new_traj.add(cx, cy, frame_idx)
                self.active[self.next_id] = new_traj
                self.next_id += 1

        # 移除未匹配的活跃轨迹
        for tid in list(self.active.keys()):
            if tid not in used:
                self.trajectories.append(self.active.pop(tid))

    def finalize(self):
        """处理结束：把剩余的活跃轨迹归档"""
        for tid in list(self.active.keys()):
            self.trajectories.append(self.active.pop(tid))
        # 过滤太短的轨迹
        self.trajectories = [t for t in self.trajectories if len(t) >= 3]
        return self.trajectories

    def update_with_region(self, region: Dict, frame_idx: int):
        """
        轻量版 update：直接接收已提取的质心数据（光流追踪模式无需 mask+flow）。
        region = {"centroid": np.array([x,y]), "area": float, ...}
        """
        cx, cy = region["centroid"]

        # 最近邻关联（与 update 相同逻辑）
        best_id = None
        best_dist = float("inf")
        for tid, traj in self.active.items():
            if traj.last_point is None:
                continue
            lx, ly = traj.last_point
            dist = np.hypot(cx - lx, cy - ly)
            if dist < 80 and dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is not None:
            self.active[best_id].add(cx, cy, frame_idx)
        else:
            new_traj = Trajectory()
            new_traj.add(cx, cy, frame_idx)
            self.active[self.next_id] = new_traj
            self.next_id += 1


# ===================================================================
# 5. 轨迹特征提取
# ===================================================================
class TrajectoryFeatureExtractor:
    """
    计算每条轨迹的：
      - 曲率 (curvature)
      - 闭合度 (closure)
      - 运动方向 (direction)
      - 速度统计 (speed_mean, speed_max, speed_std)
    """

    @staticmethod
    def curvature(points: np.ndarray) -> np.ndarray:
        """
        离散曲率： κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        points: (N, 2)
        """
        if len(points) < 3:
            return np.array([0.0])
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        num = np.abs(dx * ddy - dy * ddx)
        den = (dx ** 2 + dy ** 2) ** 1.5 + 1e-8
        return num / den

    @staticmethod
    def closure(points: np.ndarray) -> float:
        """
        闭合度：起点到终点的距离 / 轨迹总弧长。 值越小越闭合。
        """
        if len(points) < 2:
            return 1.0
        start_end_dist = np.linalg.norm(points[-1] - points[0])
        total_arc = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        if total_arc < 1e-8:
            return 1.0
        return float(start_end_dist / total_arc)

    @staticmethod
    def direction_stats(flow: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
        """
        光流主方向统计（基于全图光流 ROI 区域）。
        返回： mean_angle (弧度), consistency (0~1，方向一致性)
        """
        if mask is not None:
            valid = mask.astype(bool)
        else:
            valid = np.ones(flow.shape[:2], dtype=bool)

        fx = flow[..., 0][valid]
        fy = flow[..., 1][valid]
        mag = np.sqrt(fx ** 2 + fy ** 2)
        if len(mag) == 0:
            return {"mean_angle": 0.0, "consistency": 0.0}

        # 仅取运动显著的点
        threshold = np.percentile(mag, 70)
        significant = mag > max(threshold, 1.0)
        if not np.any(significant):
            return {"mean_angle": 0.0, "consistency": 0.0}

        angles = np.arctan2(fy[significant], fx[significant])
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        consistency = np.abs(np.mean(np.exp(1j * angles)))
        return {"mean_angle": float(mean_angle), "consistency": float(consistency)}

    @staticmethod
    def speed_stats(flow: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
        """速度统计"""
        if mask is not None:
            valid = mask.astype(bool)
        else:
            valid = np.ones(flow.shape[:2], dtype=bool)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mag_valid = mag[valid]
        if len(mag_valid) == 0:
            return {"speed_mean": 0.0, "speed_max": 0.0, "speed_std": 0.0}
        return {
            "speed_mean": float(np.mean(mag_valid)),
            "speed_max": float(np.max(mag_valid)),
            "speed_std": float(np.std(mag_valid)),
        }

    @classmethod
    def extract_trajectory_features(cls, traj: Trajectory) -> Dict:
        """对单条轨迹提取综合特征"""
        pts = np.array(traj.points)
        curv = cls.curvature(pts)
        return {
            "num_points": len(pts),
            "curvature_mean": float(np.mean(curv)),
            "curvature_max": float(np.max(curv)),
            "curvature_std": float(np.std(curv)),
            "closure": cls.closure(pts),
            "total_arc": float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))),
            "velocity_mean": float(np.mean(traj.velocities)) if traj.velocities else 0.0,
            "velocity_max": float(np.max(traj.velocities)) if traj.velocities else 0.0,
            "direction_mean": float(np.mean(traj.directions)) if traj.directions else 0.0,
        }


# ===================================================================
# 6. 时序平滑（卡尔曼滤波 + 滑动窗口）
# ===================================================================
class KalmanSmoother:
    """
    对 2D 轨迹点做卡尔曼滤波平滑。
    状态向量: [x, y, vx, vy]
    观测向量: [x, y]
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 1e-2, measure_noise: float = 1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        self.initialized = False

    def smooth(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """对整条轨迹做前向-后向平滑"""
        if len(points) < 2:
            return np.array(points, dtype=np.float32)
        pts = np.array(points, dtype=np.float32)
        # 前向
        forward = []
        # OpenCV 5 要求 state 为 (N, 1) 二维形状
        self.kf.statePre = np.array([[pts[0][0]], [pts[0][1]], [0], [0]], np.float32)
        self.kf.statePost = np.array([[pts[0][0]], [pts[0][1]], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        for pt in pts:
            pred = self.kf.predict()
            meas = np.array([[pt[0]], [pt[1]]], np.float32)
            est = self.kf.correct(meas)
            forward.append([float(est[0, 0]), float(est[1, 0])])
        forward = np.array(forward)

        # 后向
        backward = []
        self.kf.statePre = np.array([[pts[-1][0]], [pts[-1][1]], [0], [0]], np.float32)
        self.kf.statePost = np.array([[pts[-1][0]], [pts[-1][1]], [0], [0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        for pt in reversed(pts):
            self.kf.predict()
            meas = np.array([[pt[0]], [pt[1]]], np.float32)
            est = self.kf.correct(meas)
            backward.append([float(est[0, 0]), float(est[1, 0])])
        backward = np.array(backward[::-1])

        # 融合
        return (forward + backward) / 2.0

    @staticmethod
    def sliding_window_smooth(points: np.ndarray, window: int = 5) -> np.ndarray:
        """滑动窗口均值平滑"""
        if len(points) < window:
            return points
        smoothed = points.copy().astype(np.float64)
        for i in range(points.shape[1]):
            smoothed[:, i] = savgol_filter(points[:, i], min(window, len(points) - (len(points) % 2 == 0)), 2)
        return smoothed


# ===================================================================
# 7. 可视化
# ===================================================================
class Visualizer:
    """绘制光流、轨迹、光弧叠加图"""

    @staticmethod
    def draw_trajectories(frame: np.ndarray, trajectories: List[Trajectory],
                          smoothed: bool = False, smoother: KalmanSmoother = None) -> np.ndarray:
        canvas = frame.copy()
        colors = [
            (0, 255, 255),   # 青
            (0, 165, 255),   # 橙
            (255, 0, 255),   # 品红
            (0, 255, 0),     # 绿
            (255, 255, 0),   # 天蓝
        ]
        for i, t in enumerate(trajectories):
            color = colors[i % len(colors)]
            pts = np.array(t.points, dtype=np.float32)
            if smoothed and smoother is not None and len(t) >= 3:
                pts = smoother.smooth(t.points)
                pts = pts.astype(np.float32)
            pts_int = pts.astype(np.int32)
            # 画轨迹线
            for j in range(1, len(pts_int)):
                cv2.line(canvas, tuple(pts_int[j - 1]), tuple(pts_int[j]), color, 2, cv2.LINE_AA)
            # 画点
            for p in pts_int:
                cv2.circle(canvas, tuple(p), 3, color, -1, cv2.LINE_AA)
            # 起点/终点标记
            if len(pts_int) > 0:
                cv2.circle(canvas, tuple(pts_int[0]),  6, (255, 255, 255), -1)
                cv2.circle(canvas, tuple(pts_int[-1]), 6, (0, 0, 255), -1)
        return canvas

    @staticmethod
    def draw_light_arc(frame: np.ndarray, trajectories: List[Trajectory],
                       tail: int = 30, smoother: KalmanSmoother = None) -> np.ndarray:
        """
        光弧叠加图：最近 N 帧轨迹叠加在同一帧上，模拟长时间曝光效果。
        """
        canvas = np.zeros_like(frame)
        colors = [
            (0, 255, 255), (0, 165, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
        ]
        for i, t in enumerate(trajectories):
            color = colors[i % len(colors)]
            pts = np.array(t.points[-tail:], dtype=np.float32)
            if smoother is not None and len(pts) >= 3:
                pts = smoother.smooth([tuple(p) for p in pts])
            pts_int = pts.astype(np.int32)
            for j in range(1, len(pts_int)):
                alpha = 0.3 + 0.7 * j / max(len(pts_int) - 1, 1)
                cv2.line(canvas, tuple(pts_int[j - 1]), tuple(pts_int[j]), color, 3, cv2.LINE_AA)
                # 光晕叠加
                overlay = frame.copy()
                cv2.line(overlay, tuple(pts_int[j - 1]), tuple(pts_int[j]),
                         (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha)), 8, cv2.LINE_AA)
                canvas = cv2.addWeighted(canvas, 0.8, overlay, 0.2, 0)
        return cv2.addWeighted(frame, 0.4, canvas, 0.6, 0)

    @staticmethod
    def draw_flow(flow_vis: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """光流可视化，叠加 mask 边界"""
        result = flow_vis.copy()
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 255), 1)
        return result


# ===================================================================
# 8. 主流程 Pipeline
# ===================================================================
class WotaOpticalFlowPipeline:
    """
    一键式 Wota艺光流提取流水线。
    """

    def __init__(self, video_path: str, method: str = "farneback",
                 color_presets: List[str] = None, device: str = "cuda",
                 mode: str = "optical_flow"):
        """
        mode: "optical_flow" (稠密光流) 或 "light_tracking" (质心追踪，极速)
        """
        self.reader = VideoReader(video_path)
        self.masker = ROIMasker(color_presets=color_presets or ["white"])
        self.flow_engine = OpticalFlowEngine(method=method, device=device)
        self.tracker = TrajectoryTracker(max_disappear=5, min_area=30)
        self.smoother = KalmanSmoother()
        self.feature_extractor = TrajectoryFeatureExtractor()
        self.mode = mode

        # 结果存储
        self.flow_frames: List[np.ndarray] = []
        self.flow_vis_frames: List[np.ndarray] = []
        self.trajectories: List[Trajectory] = []
        self.frame_features: List[Dict] = []  # 每帧的光流特征

    def run(self, step: int = 1, max_frames: int = 300, output_dir: str = "./output"):
        """
        执行完整流水线。
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        frames = self.reader.extract(step=step, max_frames=max_frames)
        print(f"共抽取 {len(frames)} 帧 (step={step})")

        if len(frames) < 2:
            raise ValueError("帧数不足，至少需要 2 帧")

        # 自适应：若未指定色域，从第一帧采样
        if not self.masker.masks:
            self.masker = ROIMasker.from_sample(frames[0])
            print("已自适应采样光棒色彩阈值")

        if self.mode == "light_tracking":
            self._run_light_tracking(frames, output_dir)
        else:
            self._run_optical_flow(frames, output_dir)

        # 轨迹归档 + 特征
        self.trajectories = self.tracker.finalize()
        print(f"共提取 {len(self.trajectories)} 条轨迹")

        for t in self.trajectories:
            feats = self.feature_extractor.extract_trajectory_features(t)
            t.features = feats  # 动态附加

        # 输出轨迹特征汇总
        for i, t in enumerate(self.trajectories):
            print(f"  轨迹 {i+1}: {len(t)} 点, "
                  f"曲率均值={t.features['curvature_mean']:.3f}, "
                  f"闭合度={t.features['closure']:.3f}, "
                  f"速度均值={t.features['velocity_mean']:.2f}")

        # ---- 可视化输出 ----
        self._save_visualizations(frames, output_dir)
        return self

    # ================================================================
    # 稠密光流模式（原有逻辑，已提取为子方法）
    # ================================================================
    def _run_optical_flow(self, frames, output_dir):
        """稠密光流管线：Farneback / RAFT"""
        prev_frame = frames[0]
        prev_mask = self.masker.apply(prev_frame)

        for i in tqdm(range(1, len(frames)), desc="光流计算"):
            curr_frame = frames[i]
            curr_mask = self.masker.apply(curr_frame)

            # 合并两帧的 mask（取并集）
            combined_mask = cv2.bitwise_or(prev_mask, curr_mask)

            # 光流
            flow, flow_vis = self.flow_engine.compute(prev_frame, curr_frame, combined_mask)
            self.flow_frames.append(flow)
            self.flow_vis_frames.append(flow_vis)

            # 轨迹追踪
            self.tracker.update(curr_mask, flow, i)

            # 帧级光流特征
            feat = {}
            feat.update(self.feature_extractor.direction_stats(flow, combined_mask))
            feat.update(self.feature_extractor.speed_stats(flow, combined_mask))
            feat["frame_idx"] = i
            self.frame_features.append(feat)

            prev_frame = curr_frame
            prev_mask = curr_mask

    # ================================================================
    # 质心追踪模式 —— 极速版，跳过稠密光流
    # ================================================================
    def _run_light_tracking(self, frames, output_dir):
        """
        不计算稠密光流，直接通过颜色蒙版追踪光棒质心。
        速度提升 10~50 倍，适用于 Wota艺 等高对比度场景。
        """
        print(f"[极速模式] 质心追踪 — 跳过稠密光流计算")
        self.flow_frames = [np.zeros((1, 1, 2), dtype=np.float32)]  # 占位

        # 存储质心序列用于可视化
        all_centroids = []  # List[List[Tuple[float, float]]]

        for i in tqdm(range(len(frames)), desc="质心追踪"):
            mask = self.masker.apply(frames[i])
            centroids = self._extract_centroids(mask)
            all_centroids.append(centroids)

            # 构建伪光流可视化 (每帧一张空白占位图)
            h, w = frames[i].shape[:2]
            self.flow_vis_frames.append(np.zeros((h, w, 3), dtype=np.uint8))

            if i > 0:
                prev_centroids = all_centroids[i - 1]
                # 帧级特征：基于质心位移计算方向和速度
                feat = self._compute_centroid_motion_features(prev_centroids, centroids)
                feat["frame_idx"] = i
                self.frame_features.append(feat)

            # 用 centroids 更新 tracker（伪装成"光流"区域）
            self._update_tracker_from_centroids(centroids, i)

        # 将质心序列写入轨迹的可视化数据
        self._light_centroids = all_centroids

    # ================================================================
    # 质心提取
    # ================================================================
    @staticmethod
    def _extract_centroids(mask: np.ndarray) -> List[Tuple[float, float]]:
        """
        从二值掩膜中提取每个连通区域的质心。
        返回: [(cx, cy), ...] 按面积从大到小排序
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        # 按面积排序，附带质心
        items = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                items.append((area, cx, cy))
        items.sort(key=lambda x: -x[0])  # 面积降序
        return [(cx, cy) for _, cx, cy in items[:5]]

    # ================================================================
    # 基于质心位移计算帧级特征
    # ================================================================
    @staticmethod
    def _compute_centroid_motion_features(
        prev_centroids: List[Tuple[float, float]],
        curr_centroids: List[Tuple[float, float]],
    ) -> Dict:
        """
        通过质心在帧间的位移近似光流的"方向"和"速度"。
        """
        # 最近邻匹配
        if not prev_centroids or not curr_centroids:
            return {"mean_angle": 0, "consistency": 0,
                    "speed_mean": 0, "speed_max": 0, "speed_std": 0}

        angles = []
        speeds = []
        used_curr = set()

        for pc in prev_centroids:
            best_dist = float("inf")
            best_cc = None
            for j, cc in enumerate(curr_centroids):
                if j in used_curr:
                    continue
                dist = np.hypot(pc[0] - cc[0], pc[1] - cc[1])
                if dist < best_dist and dist < 200:  # 最大匹配距离阈值
                    best_dist = dist
                    best_cc = (j, cc)
            if best_cc is not None:
                j, cc = best_cc
                used_curr.add(j)
                dx = cc[0] - pc[0]
                dy = cc[1] - pc[1]
                angle = np.arctan2(dy, dx)
                angles.append(angle)
                speeds.append(np.hypot(dx, dy))

        if not angles:
            return {"mean_angle": 0, "consistency": 0,
                    "speed_mean": 0, "speed_max": 0, "speed_std": 0}

        # 方向一致性 → 用 mean resultant length (归一化)
        cos_sum = sum(np.cos(a) for a in angles)
        sin_sum = sum(np.sin(a) for a in angles)
        consistency = np.hypot(cos_sum, sin_sum) / len(angles)
        mean_angle = np.arctan2(sin_sum, cos_sum)

        speeds_arr = np.array(speeds)
        return {
            "mean_angle": float(mean_angle),
            "consistency": float(consistency),
            "speed_mean": float(np.mean(speeds_arr)),
            "speed_max": float(np.max(speeds_arr)),
            "speed_std": float(np.std(speeds_arr)),
        }

    # ================================================================
    # 质心 → TrajectoryTracker 适配
    # ================================================================
    def _update_tracker_from_centroids(self, centroids, frame_idx):
        """
        将质心数据"喂"给现有的 TrajectoryTracker。
        TrajectoryTracker 内部用 centroid + area 做关联，
        所以我们伪造一个 min_flow_region 填入质心坐标。
        """
        for cx, cy in centroids:
            # 构造一个虚拟的"光流区域"给 tracker
            fake_region = {
                "centroid": np.array([cx, cy], dtype=np.float32),
                "area": 50,   # 伪面积
                "pixels": [],  # tracker 不需要 pixel 列表
            }
            self.tracker.update_with_region(fake_region, frame_idx)

    def _save_visualizations(self, frames, output_dir):
        vis = Visualizer()

        if self.mode == "light_tracking":
            # 极速模式：用质心轨迹替代光流视频, 输出帧叠加质心
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(f"{output_dir}/centroid_tracking.mp4", fourcc,
                                     self.reader.fps, (w, h))
            centroids = getattr(self, "_light_centroids", [])
            for i, frame in enumerate(frames):
                disp = frame.copy()
                if i < len(centroids):
                    for cx, cy in centroids[i]:
                        cv2.circle(disp, (int(cx), int(cy)), 8, (0, 255, 200), -1)
                        cv2.circle(disp, (int(cx), int(cy)), 12, (0, 255, 200), 2)
                # 绘制累积轨迹（显示完整路径）
                for t in self.trajectories:
                    pts = t.points
                    if len(pts) >= 2:
                        for k in range(1, len(pts)):
                            pt1 = (int(pts[k-1][0]), int(pts[k-1][1]))
                            pt2 = (int(pts[k][0]), int(pts[k][1]))
                            cv2.line(disp, pt1, pt2, (0, 255, 200), 2)
                writer.write(disp)
            writer.release()
            print(f"质心追踪视频已保存: {output_dir}/centroid_tracking.mp4")
        else:
            # 光流可视化视频
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(f"{output_dir}/optical_flow.mp4", fourcc,
                                     self.reader.fps, (w * 2, h))
            for i in range(len(self.flow_vis_frames)):
                flow_disp = vis.draw_flow(self.flow_vis_frames[i])
                combined = np.hstack([frames[i + 1], flow_disp])
                writer.write(combined)
            writer.release()
            print(f"光流视频已保存: {output_dir}/optical_flow.mp4")

        # (b) 轨迹叠加图
        try:
            traj_img = vis.draw_trajectories(frames[-1], self.trajectories,
                                             smoothed=True, smoother=self.smoother)
            cv2.imwrite(f"{output_dir}/trajectories.png", traj_img)
            print(f"轨迹图已保存: {output_dir}/trajectories.png")
        except Exception as e:
            print(f"[警告] 轨迹图生成失败: {e}")

        # (c) 光弧叠加图
        try:
            arc_img = vis.draw_light_arc(frames[-1], self.trajectories, tail=100, smoother=self.smoother)
            cv2.imwrite(f"{output_dir}/light_arc.png", arc_img)
            print(f"光弧图已保存: {output_dir}/light_arc.png")
        except Exception as e:
            print(f"[警告] 光弧图生成失败: {e}")

    def get_embedding_snapshot(self, normalize: bool = True) -> np.ndarray:
        """
        输出一帧的"光流特征快照"向量（用于后续 Embedding）。
        拼接：轨迹特征均值 + 末帧光流方向/速度
        """
        traj_vecs = []
        for t in self.trajectories:
            traj_vecs.append([
                t.features["curvature_mean"],
                t.features["curvature_max"],
                t.features["curvature_std"],
                t.features["closure"],
                t.features["velocity_mean"],
                t.features["velocity_max"],
                t.features["direction_mean"],
            ])
        if traj_vecs:
            traj_mean = np.mean(traj_vecs, axis=0)
        else:
            traj_mean = np.zeros(7)

        if self.frame_features:
            last_feat = self.frame_features[-1]
            frame_vec = np.array([
                last_feat["mean_angle"],
                last_feat["consistency"],
                last_feat["speed_mean"],
                last_feat["speed_max"],
                last_feat["speed_std"],
            ])
        else:
            frame_vec = np.zeros(5)

        vec = np.concatenate([traj_mean, frame_vec])
        if normalize and np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        return vec


# ===================================================================
# 9. 命令行入口
# ===================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wota艺 光流轨迹提取")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("-o", "--output", default="./output", help="输出目录")
    parser.add_argument("-m", "--method", default="farneback",
                        choices=["farneback", "raft", "light_tracking"],
                        help="光流算法 (light_tracking=极速质心追踪, farneback/raft=稠密光流)")
    parser.add_argument("--mode", default="light_tracking",
                        choices=["optical_flow", "light_tracking"],
                        help="管线模式: light_tracking=极速 (推荐), optical_flow=稠密光流")
    parser.add_argument("-s", "--step", type=int, default=1, help="抽帧步长")
    parser.add_argument("-n", "--max-frames", type=int, default=300, help="最大处理帧数")
    parser.add_argument("-c", "--colors", nargs="*",
                        default=["white"],
                        help="光棒颜色预设, 可选: orange blue red pink green white purple")
    parser.add_argument("-d", "--device", default="cuda", help="设备 (cuda/cpu)")

    args = parser.parse_args()

    pipeline = WotaOpticalFlowPipeline(
        video_path=args.video,
        method=args.method,
        color_presets=args.colors,
        device=args.device,
        mode=args.mode,
    )
    pipeline.run(step=args.step, max_frames=args.max_frames, output_dir=args.output)

    # 输出光流特征快照向量
    vec = pipeline.get_embedding_snapshot()
    print(f"\n光流特征快照向量 ({len(vec)}维):\n{vec}")
