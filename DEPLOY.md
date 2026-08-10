# WOTA艺 技术识别系统 — 部署文档

## 一、系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户浏览器                           │
│               http://127.0.0.1:5000                     │
└──────────────────────┬──────────────────────────────────┘
                       │ 上传视频 / 轮询结果
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask Web 后端 (app.py)                │
│                                                         │
│  ① 接收视频 → ② 光流提取 → ③ 向量检索 → ④ 返回结果        │
└───────┬─────────────────────────────────┬───────────────┘
        │                                 │
        ▼                                 ▼
┌───────────────────┐           ┌──────────────────┐
│ optical_flow_wota │           │  vector_db_wota  │
│   光流提取引擎      │ ──12维──▶ │  Faiss 向量数据库  │
│                   │  向量      │                  │
│ · 抽帧             │           │ · IndexFlatIP    │
│ · ROI掩膜          │           │ · IndexIVFFlat   │
│ · RAFT/Farneback  │           │ · 余弦相似度检索   │
│ · 轨迹追踪          │           │ · 持久化存储      │
│ · 特征提取          │           │                  │
│ · 卡尔曼滤波        │           └──────────────────┘
└───────────────────┘
```

## 二、环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10 / Linux / macOS | Windows 10+ |
| Python | 3.10+ | 3.11+ |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 2 GB 可用空间 | 5 GB+ |
| GPU（可选） | 无（CPU可运行） | NVIDIA GPU + CUDA（RAFT加速） |

## 三、安装步骤

### 3.1 克隆或下载项目

```bash
# 项目目录结构
项目根目录/
├── app.py                  # Flask Web 后端
├── optical_flow_wota.py    # 光流提取引擎
├── vector_db_wota.py       # Faiss 向量数据库
├── templates/
│   └── index.html          # Web 前端页面
├── static/                 # 静态文件（自动创建）
├── uploads/                # 用户上传暂存（自动创建）
└── wota_db.index           # 向量数据库索引（入库后生成）
    wota_db.meta            # 向量数据库元数据（入库后生成）
```

### 3.2 安装 Python 依赖

```bash
# 核心依赖（必须安装）
pip install opencv-python    # 视频处理 + 光流计算
pip install numpy            # 数值计算
pip install scipy            # 信号滤波
pip install flask            # Web 服务

# 向量数据库（必须安装）
pip install faiss-cpu        # CPU 版 Faiss（GPU 版：faiss-gpu）

# 可选依赖
pip install tqdm             # 进度条显示
pip install torch torchvision  # RAFT 光流（需 GPU，CPU 可跳过）
```

> **一键安装：**
> ```bash
> pip install opencv-python numpy scipy flask faiss-cpu tqdm
> ```

## 四、使用流程

### 4.1 第一步：录入标准技术（建立数据库）

将已知技术的标准视频录入向量数据库，AI 才能进行比对识别。

```bash
# 基础入库
python vector_db_wota.py add ./videos/sandaa_sunneku.mp4 -n "サンダースネーク" -c "雷蛇"

# 指定光棒颜色（提高 ROI 精度）
python vector_db_wota.py add ./videos/romance.mp4 -n "ロマンス" -c "ロマンス" --colors orange cyan

# 参数说明
#   video         标准视频文件路径
#   -n, --name    技术名称
#   -c, --category 技术分类（如：技、雷蛇、ロマンス、虽然等）
#   -d, --db      数据库路径前缀（默认 ./wota_db）
#   --colors      光棒颜色预设（默认 bright）
#                 可选: cyan red orange pink green bright
#   --target-dim  投影目标维度（默认 512）
#   -n, --max-frames 最大处理帧数（默认 300）
```

**批量入库示例：**

```bash
# 逐条录入多个技术
python vector_db_wota.py add ./std/OAD.mp4        -n "OAD"            -c "技"    --colors cyan
python vector_db_wota.py add ./std/romance.mp4    -n "ロマンス"        -c "ロマンス" --colors orange
python vector_db_wota.py add ./std/sandaa.mp4     -n "サンダースネーク"  -c "雷蛇"   --colors cyan orange
python vector_db_wota.py add ./std/rouman.mp4     -n "羅曼"           -c "技"    --colors red
python vector_db_wota.py add ./std/hanaichimonme.mp4 -n "花一匁"      -c "技"    --colors pink
```

### 4.2 第二步：查看数据库状态

```bash
# 列出所有已录入技术
python vector_db_wota.py list -d ./wota_db

# 输出示例：
# 数据库: 5 条技术
# ----------------------------------------
#   [a1b2c3d4e5f6] サンダースネーク        雷蛇
#   [b2c3d4e5f6a7] ロマンス              ロマンス
#   [c3d4e5f6a7b8] OAD                   技
#   ...
```

### 4.3 第三步：启动 Web 服务

```bash
python app.py
```

启动成功后会显示：

```
数据库已加载: 5 条技术

访问 http://127.0.0.1:5000
```

### 4.4 第四步：使用 Web 界面

1. 浏览器打开 `http://127.0.0.1:5000`
2. 页面顶部会显示数据库状态（已录入 X 个技术 或 未初始化）
3. 点击上传区域或拖拽视频文件
4. 选择光棒颜色（根据视频中实际使用的光棒颜色）
5. 点击「开始识别」
6. 等待处理完成（通常 10~60 秒，取决于视频长度和帧数）
7. 查看识别结果：
   - 技术名称 + 置信度百分比
   - Top-5 候选排名列表
   - 轨迹追踪图、光弧叠加图、光流视频

## 五、命令行速查

### 光流提取（单独使用）

```bash
# 仅提取光流特征，不检索数据库
python optical_flow_wota.py video.mp4 -o ./output

# 参数说明
#   video         输入视频路径
#   -o, --output  输出目录（默认 ./output）
#   -m, --method  光流算法 farneback（CPU）或 raft（GPU）
#   -s, --step    抽帧步长（默认1，每帧都处理）
#   -n, --max-frames 最大处理帧数（默认300）
#   -c, --colors  光棒颜色预设（默认 bright）
#   -d, --device  设备 cuda/cpu（默认 cuda）

# 输出文件：
#   output/optical_flow.mp4  光流可视化视频
#   output/trajectories.png  轨迹追踪图
#   output/light_arc.png     光弧叠加图
```

### 向量检索（命令行）

```bash
# 命令行搜索（用于调试）
python vector_db_wota.py search ./user_video.mp4 -k 5 --min-score 0.5

# 输出示例：
# =======================================================
#   查询视频: ./user_video.mp4
# =======================================================
#   1. サンダースネーク        [雷蛇]
#      置信度: 87.32%  ██████████████████████████
#      标准视频: ./videos/sandaa_sunneku.mp4
```

### 删除技术

```bash
# 根据 move_id 删除
python vector_db_wota.py delete a1b2c3d4e5f6 -d ./wota_db
```

## 六、配置参数

| 配置项 | 位置 | 默认值 | 说明 |
|--------|------|--------|------|
| 服务端口 | `app.py` L246 | `5000` | 修改 `port=5000` |
| 上传大小限制 | `app.py` L34 | `200MB` | 修改 `MAX_CONTENT_LENGTH` |
| 数据库路径 | `app.py` L37 | `./wota_db` | 环境变量 `WOTA_DB_PATH` |
| 最大处理帧数 | `app.py` L76 | `300` | 修改 `max_frames` |
| ROI 最小面积 | `optical_flow_wota.py` L263 | `30` 像素 | 修改 `min_area` |
| 轨迹关联距离阈值 | `optical_flow_wota.py` L305 | `80` 像素 | 修改 `dist < 80` |
| Faiss 索引策略 | `vector_db_wota.py` L134 | `auto` | 修改 `index_type` |
| IVF 聚类数 | `vector_db_wota.py` L114 | `100` | 修改 `nlist` |

## 七、常见问题

### Q1: 识别结果置信度很低怎么办？

1. 检查光棒颜色选择是否与视频匹配
2. 尝试组合多种颜色（如 `--colors cyan orange pink`）
3. 确保标准视频的光流提取参数与查询视频一致
4. 增加 `max_frames` 让系统处理更多帧

### Q2: 处理速度太慢？

1. 减少 `max_frames`（如设为 150）
2. 增大抽帧步长 `-s 2`（隔帧处理）
3. 使用 `-m farneback` 代替 `-m raft`（CPU 友好）
4. 安装 `faiss-gpu` 加速向量检索

### Q3: 数据库为空如何测试？

即使没有标准视频入库，系统仍可运行。光流提取和可视化功能正常，但不会返回技术匹配结果（显示"数据库中未找到匹配的技术"）。

### Q4: 如何升级到 A路骨骼 + B路光流 融合？

系统已预留接口：

```python
from vector_db_wota import FeatureProjector

# 假设骨骼模块输出 512 维向量 skeleton_vec，光流模块输出 512 维 optical_vec
combined = FeatureProjector.concatenate(skeleton_vec, optical_vec)  # → 1024 维

# 入库时直接使用拼接后的向量
record = MoveRecord(move_name="...", vector=combined)
db.insert(record)
```

### Q5: 能否部署到公网？

Flask 开发服务器仅适用于本地测试。生产环境建议：

```bash
# 方案一：使用 gunicorn（Linux/macOS）
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 方案二：Nginx 反向代理 + gunicorn
# 方案三：Docker 容器化部署
```

## 八、技术栈

| 层级 | 技术 |
|------|------|
| 前端 | HTML5 + CSS3 + JavaScript（原生） |
| 后端 | Python 3.10+ / Flask 3.x |
| 光流算法 | RAFT（torchvision）/ Farneback（OpenCV） |
| 向量数据库 | Faiss（IndexFlatIP / IndexIVFFlat） |
| 数值计算 | NumPy / SciPy |
| 时序滤波 | 卡尔曼滤波（OpenCV）/ Savitzky-Golay（SciPy） |
| 可视化 | OpenCV（绘图）/ Matplotlib（可选） |
