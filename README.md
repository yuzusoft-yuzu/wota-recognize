---
title: Wota艺 动作识别系统
emoji: 🎵
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 5000
pinned: false
---

# Wota艺 动作识别系统（骨光融合版）

基于 **「骨光融合」** 的 Wota艺（ヲタ芸）动作识别网站：同时提取**人体骨骼**与**光棒轨迹**，
经 DTW 时序比对，识别用户上传视频对应的「技名」并给出匹配度。

## 三大板块（多页面）

| 板块 | 路由 | 面向 | 作用 |
|------|------|------|------|
| 动作识别 | `/recognize` | 用户 | 上传视频 → 比对数据库 → 返回技名 + 匹配度 |
| 技术总览 | `/techniques` | 用户/管理员 | 搜索引擎浏览技名、查看详情；管理员可修改（日语名、B站链接等） |
| 管理员界面 | `/admin` | 管理员 | 账号密码登录 → 上传标准动作视频、标注技名 → 分析入库 |

门户首页 `/` 提供三个板块入口。技术总览的技名**均来自管理员上传的视频，初始为空**。

## 核心业务逻辑

### 骨光融合提取（核心）
- **A路 骨骼**：MediaPipe Pose → 提取手腕等关键关节坐标（未安装 mediapipe 时降级为纯光棒模式）。
- **B路 光棒**：OpenCV（HSV 颜色过滤 / 亮度阈值）→ 提取光斑质心。
- **ROI 优化**：利用骨骼「手腕」坐标，仅在手腕周围划定 ROI 内寻找最亮光斑，排除背景干扰、省去复杂追踪。
- **暗光兜底**：送入模型前对 L 通道做 CLAHE 自适应直方图均衡化，缓解会场过暗导致骨骼丢失。

### 长视频处理与抽帧
- **动态抽帧**：每 2 帧取 1 帧 + 「变化像素占比」检测，静止帧跳过（占比法避免暗背景下小光棒被误判为静止）。
- **切片 → 并行推理 → 结果聚合**：帧序列切片后用线程池并行提取特征，按时序拼接 + 滑动均值/Savitzky-Golay 时序平滑。

### 相似度比对
- 每帧融合为固定 28 维特征（骨骼 18 维 + 光棒 10 维）。
- **DTW（动态时间规整）** 带状约束（Sakoe-Chiba）做时序对齐，相似度 = `exp(-avg_dist/sigma)`。
- 识别聚合：全序列 DTW 相似度（主） + 切片投票（投票/时序平滑），综合排序输出 Top-K 与匹配度。

## 目录结构

```
wota-system/
├── app.py                   # Flask 后端：3 页面 + 鉴权 + 识别/入库/检索 API
├── skeleton_light_fusion.py # 核心：骨光融合提取 + 切片并行 + DTW + 识别器
├── wota_database.py         # SQLite 技术库（技名/日语名/B站/描述 + 逐帧特征序列）+ 管理员账号
├── templates/
│   ├── base.html            # 共享导航布局
│   ├── portal.html          # 门户首页
│   ├── recognize.html       # 动作识别
│   ├── techniques.html      # 技术总览
│   └── admin.html           # 管理员界面
├── static/
│   ├── css/style.css
│   └── js/{recognize,techniques,admin}.js
├── uploads/                 # 上传暂存（自动）
├── static/output/           # 预览图（自动）
└── wota_tech.db             # SQLite 数据库（自动创建，初始为空）
```

> `optical_flow_wota.py` / `vector_db_wota.py` 为旧版光流+Faiss 实现，已不在新流程中使用，保留供参考。

## 安装与运行

```bash
pip install -r requirements.txt
# 可选（启用骨骼融合）：
# pip install mediapipe

python app.py
# 访问 http://127.0.0.1:5000
```

### 管理员账号
默认 `admin` / `admin123`，可通过环境变量覆盖：
```bash
set ADMIN_USER=admin
set ADMIN_PASSWORD=你的密码
```

## 使用流程

1. **管理员** 进入「管理员界面」登录，上传约 10 秒、≤50MB 的标准动作视频并标注技名（可填日语名、分类、B站链接、描述）。系统分析后入库。
2. **用户** 在「动作识别」上传视频，获得技名与匹配度；在「技术总览」搜索浏览技名详情。
3. **管理员** 在「技术总览」点击技名可修改日语名、B站链接等详细信息或删除。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `PORT` | 5000 | 服务端口 |
| `ADMIN_USER` | admin | 管理员账号 |
| `ADMIN_PASSWORD` | admin123 | 管理员密码 |
| `WOTA_DB_PATH` | ./wota_tech.db | 数据库路径 |
| `SECRET_KEY` | 随机 | Flask 会话密钥 |
