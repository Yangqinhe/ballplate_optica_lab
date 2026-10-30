# 核心源码导航

这里只归档源码，数据和权重由实验机单独管理。路径中 `experiments/E…` 表示代码来源，目录内不包含该实验的采样数据。

## 可独立使用的运算

- `src/optical_lab/control.py`：64 区积分、64→2 仿射头、四时刻历史、冲量换算、独立采集窗。由 E107 提取原函数体，不导入 SDK。
- `src/optical_lab/effective_frame.py`：独立有限窗均值 / EMA；每窗完成后清空历史。
- `src/optical_lab/regions.py`：8/16/32/64 区嵌套划分，ROI 与分区覆盖检查。
- `src/optical_lab/metrics.py`、`readout.py`：模型/实测比较和读出计算。`readout.py` 中的历史预处理示例不是当前控制链的默认参数。
- 其余 `src` 模块保留历史离线研究工具；不应把它们全部当作当前在线控制依赖。

## 当前上位机与物理＋误差模型

以下相对路径都位于 `snapshots/workspace/ballplate_optical_lab/experiments/`：

| 入口 | 作用 |
| --- | --- |
| `E110_adjustable_settle_20261030_1/run.py`、`page.py` | 当前控制对比入口，10–1000 ms 等待控件 |
| `E109_hold_effective_frame_20261030_1/page.py` | 采样期间保持上一完整结果的显示逻辑 |
| `E107_paired_control_20261030_1/core.py`、`worker.py`、`page.py` | 两侧控制、MuJoCo 工作进程与显示 |
| `E106_surrogate_preview_20261030_1/` | 物理＋误差模型的 CCD 预览 |
| `E105_rebuilt_surrogate_20261029_1/` | 有限物理校正、残差/响应预测、噪声估计和训练代码 |
| `E081_readout_regions_20261028_1/staged/optical_lab/` | 主界面共用分区/滤波、稳定性分析 |

目录保留原依赖关系，不在本次上传中继续增加界面继承或搬迁线上入口。它们依赖未上传的本地配置和模型文件，不能仅凭代码归档直接启动实物。

## 数值仿真与驱动参考

以下路径位于 `snapshots/workspace/`：

- `ballplate_fourframe96_deploy/route1_fourier_v4/`：数值 4f 模型、训练、评估和精度审计。该训练入口明确为非 PPO，不因其他目录名含 PPO 而改变算法含义。
- `ballplate_fourframe96_deploy/`：96 像素历史输入、仿真上位机及实际平台接口参考。
- `ballplate_fourframe_optical_ppo/` 等相关目录：源代码依赖，不表示各版本权重可混用。
- `ballplate_minframes_optical_ppo/SDON_Linux_Handoff_20261015/`：平台显示、采集和校准接口源码；不包含厂商 SDK、校准数组或运行数据。
- `simulation/model.xml` 位于仓库根目录，是当前球板 MuJoCo 模型的独立副本。

不要对整个 `snapshots/` 运行递归自动测试：其中有历史维护和硬件操作脚本。默认测试范围仅为仓库根目录的 `tests/`。
