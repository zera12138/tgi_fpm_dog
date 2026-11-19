# BPX-TGI-FPM

把 [Terrain-Grounded Imagination (TGI)](https://github.com/leggedrobotics/terrain-grounded-imagination)
从 Unitree G1 人形迁移到 **BPX** 12 自由度四足的第一版适配。

策略不直接规划"最优落脚点"：FPM（foothold prediction model）从本体感知和速度指令预测
**当前策略接下来会踩到哪里**，策略只读取四个预测足点周围的局部高程块。世界模型、
想象内策略优化、SGS 地形课程和在线落足监督沿用 TGI。


## 上游与致谢

本工程建立在这些 MIT 开源工作之上，代码与设计大量复用，在此致谢：

- **[Terrain-Grounded Imagination](https://github.com/leggedrobotics/terrain-grounded-imagination)**
  （Heyrman & Hutter，ETH RSL）——世界模型、地形高程的可微采样、想象内策略梯度、
  SGS 课程、FPM 在线监督、训练主循环与导出。`tgi/` 的骨架来自它。
- **[Watch Your Step](https://leggedrobotics.github.io/watch-your-step/) /
  [foothold-prediction-model](https://github.com/leggedrobotics/foothold-prediction-model)**
  ——FPM 网络与落足标签、G1 任务包和地形生成。`fpm/` 即该任务包。
- **[quadruped_lab](https://github.com/kkmjj0721/quadruped_lab)** ——
  MoE 8 专家本体历史编码器（`tgi/src/tgi/world_model/moe.py` 改写自其
  `agents/supermimic/policy.py`），许可证见 `licenses/quadruped_lab_MIT.txt`。

BPX 的 URDF、网格与相机支架由项目方提供。上游许可证与引用信息保留在
`LICENSE`、`CITATION.cff` 和 `licenses/` 中。

## 架构

```text
tgi/                        训练与部署核心（TGI 改造）
  src/tgi/robot.py          BPX 常量：关节/足端顺序、MJCF/USD 路径、PD、默认姿态、接口版本
  src/tgi/env/              Isaac Lab 3 + Newton 环境：bpx_cfg、batched sim、SGS 地形课程
  src/tgi/mdp/              可微运动学、heightmap 采样、观测/奖励/终止、历史与任务记忆
  src/tgi/world_model/      地形条件世界模型、回放缓存、MLP/MoE 策略网络
  src/tgi/foot_pred/        落足标签、FPM 网络与在线训练
  src/tgi/training/         采集、世界模型/策略优化、日志、命令行入口
  src/tgi/deploy/           ONNX/NumPy 控制器（sim2sim 与部署共用）
fpm/                        上游 Watch Your Step 任务包：G1 环境、地形、FPM 实现
urdf/                       BPX URDF 与网格（相机支架 v9 由 bpx.urdf 引用）
tgi/src/tgi/assets/         由 URDF 生成的 MJCF/USD，以及 G1 参考资产
scripts/                    预检、CPU 回归、Newton 冒烟、资产转换、评估、导出检查
sim2sim_level7/             MuJoCo 第 7 级场景闭环 sim2sim（手柄/键盘、FPM 标记）
run.sh                      统一入口（隔离 ROS 的 LD_LIBRARY_PATH）
```

一次训练迭代的数据流：

1. 并行 BPX 环境按当前策略采集转移，世界模型学习 `(状态, 动作, 足下地形) → 下一状态/接触`；
2. FPM 用仿真里下一次真实落地做监督，在线更新；
3. 策略在世界模型里想象 32 步（每步 20 ms），把回报反传到 actor，critic 负责 bootstrap；
   想象足点落在支撑不足的地形上时截断梯度；
4. SGS 课程根据成功率调整地形采样，逐步加难。

### 接口约定

| 项目 | BPX |
|---|---|
| 关节顺序 | FL、FR、HL、HR；每腿 hip_roll、hip_pitch、knee，共 12 |
| 足端顺序 | FL、FR、HL、HR；参考点为 toe 圆柱中心 |
| 世界状态 | 37：位置 3 + wxyz 四元数 4 + 世界线速度 3 + 世界角速度 3 + q 12 + qd 12 |
| FPM 输入 / 输出 | 36（机体系线速度、角速度、重力、速度指令各 3 + 相对关节角 12 + 关节速度 12）/ 4 × 2 |
| FPM 语义 | 各足下一次落地时，足端参考点在**当前机体系**的 xy |
| 策略输入 | 48 本体状态 + 4 × (xy 2 + 7×7 高度块) = 252 |
| 动作 | 12；目标角 `q0 + 0.25 × action`，控制周期 20 ms |
| 接触预测 | 世界模型输出四足接触概率 |

## 开始

### 1. 环境

- Linux x86_64，NVIDIA GPU（CPU 可跑检查与 MuJoCo 评估）；
- Isaac Lab 3.0.0（Isaac Sim 6.1，Newton 后端）及对应 PyTorch；
- MuJoCo ≥ 3.3、onnx/onnxruntime、matplotlib（`tgi/pyproject.toml`）；
- Python ≥ 3.10。

`run.sh` 默认使用 `/home/odroid/anaconda3/envs/bpx_supermimic/bin/python`，换机器时覆盖：

```bash
export BPX_PYTHON=/path/to/isaaclab/python
```

仓库通过 `PYTHONPATH` 直接引用 `tgi/src` 与 `fpm/src`，不需要先 `pip install`。
`run.sh` 会清空 `LD_LIBRARY_PATH`，避免误导入系统里其他的 G1/ROS 包。

### 2. 预检与回归

```bash
git clone https://gitee.com/zera12138/tgi_fpm_dog.git
cd tgi_fpm_dog
export BPX_PYTHON=/path/to/isaaclab/python

./run.sh preflight          # 只读报告：Python/包/CUDA/资产路径
./run.sh check              # CPU 回归：FK、标签、镜像、回合边界、导出
./run.sh newton --device cpu   # Isaac Lab/Newton 2 环境冒烟
```

### 3. 训练

先跑 5 次迭代确认数据、损失和检查点写入正常：

```bash
./run.sh train --num_envs 32 --iterations 5 --warmup_steps 16 \
  --wm_batch 32 --wm_batches 1 --wm_hidden 128 128 --pi_batch 32 \
  --compile 0 --cudagraphs 0 --async_collect 0 --amp none --vram_gb 6 \
  --log_every 1 --ckpt_every 5 --log_dir logs/gpu_smoke
```

正式训练（默认 100000 次迭代；默认世界模型预热 300 次、FPM 在 1500 次后接管）：

```bash
./run.sh train --num_envs 1024 --vram_gb 10 \
  --compile 0 --cudagraphs 0 --async_collect 0 \
  --log_dir logs/bpx_first
```

显存足够时可增大 `--num_envs`，默认配置按 8192 环境设计。不要用小网络的启动检查点去续接
默认大网络。可选的 MoE 策略与任务记忆：

```bash
bash run_moe.sh --dry-run    # 查看续训参数；去掉 --dry-run 实际运行
```

### 4. 评估与导出

```bash
./run.sh evaluate --ckpt logs/bpx_first/last.pt --seconds 10 --cmd 0.3 0 0
./run.sh export   --ckpt logs/bpx_first/last.pt --out sim2sim_level7
```

`evaluate` 目前只做 MuJoCo 平地闭环，不代表论文地形基准。`export` 要求 BPX 检查点且 FPM
已启用，输出到 `<out>/exported/`（actor ONNX、FPM ONNX、TorchScript 与 `deploy.json`）。
导出的 rsl 格式只提供 actor 兼容布局，附带的 critic 是占位初始化，不能用于续训。

### 5. sim2sim（MuJoCo 第 7 级场景）

先按上一步把 `exported/` 生成到 `sim2sim_level7/`，再：

```bash
cd sim2sim_level7
bash run.sh
```

需要带 MuJoCo、onnxruntime、pygame 的 Python（脚本默认
`/home/odroid/anaconda3/envs/rl_lab/bin/python`）。`controls.json` 里的 `project`、
`python`、`reference_helpers` 是机器相关路径，换机器时需修改。手柄不可用时可用键盘，
键位与操作见脚本启动后的提示。

### 6. 可选：重建 MJCF/USD 资产

```bash
./run.sh assets
```

流程为 URDF → MJCF → Isaac MJCF importer → USD，保留质量、惯量和几何。USD 顶层文件不能
单独移动，否则相对 payload 引用会失效。

## 已知边界

- 高程图来自仿真全局高度图（特权信息）；尚未实现相机采集、位姿融合与历史地图，真机需要
  带有效性标记的局部历史高程图。
- 足端支撑面积使用近似采样；参考点是 toe 圆柱中心，不等于滚动接触面的瞬时接触点。


## 许可证与引用

MIT，见 `LICENSE`。若使用本工程，请同时引用上游 TGI 与 Watch Your Step，引用信息见
`CITATION.cff`。
