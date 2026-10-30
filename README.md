# Ballplate Optical Lab

基于 DMD → 4f / SLM → CCD 的光学计算球板控制研究，控制对象为 MuJoCo 球板仿真。

本仓库是 **2026-10-30 核心源码归档**。只上传代码、MuJoCo 几何模型和说明，不包含实验数据、训练权重、SLM mask、相机 SDK 或本机启动授权。原工作目录和历史实验保留在实验机，不受本次整理影响。

## 结构

```text
src/optical_lab/       离线分析、有效帧滤波、读出区域与控制运算
simulation/model.xml  MuJoCo 球板几何与动力学定义（不是网络权重）
snapshots/workspace/  按原目录保留的仿真、上位机及驱动依赖源码
tests/                不连接器件的数学/读出测试
docs/                 核心入口、运行边界与维护说明
provenance/           源文件路径、SHA256 与提取说明
```

当前链路：四幅历史二值图 → DMD 同屏输入 → SLM 光学映射 → CCD 独立 RAW 窗滤波 → 64 区积分 → 64→2 线性头 → MuJoCo。线性头为 130 参数；仿真域和真实 CCD 分别使用各自适配的头。采样时暂停 MuJoCo，每次结果完成后推进 20 ms 仿真时间，不等于现实 20 ms 实时控制。

## 离线检查

需要 Python ≥3.10。以下命令不连接相机、不投图、不启动硬件上位机：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
python run_lab.py --help
```

`run_lab.py` 保留离线研究 CLI；除 `--help` 外，涉及具体实验的命令需自行提供本地数据与配置，不能把缺失数据的命令当作可直接复现实验的演示。

## 上位机与训练入口

详见 [源码导航](docs/SOURCE_MAP.md)。当前界面源码入口为 E110，保留主界面、读出稳定性测试、仿真/真实控制对比以及 10–1000 ms 可调等待。模型、训练、MuJoCo 环境和历史上位机依赖均按原路径归档，便于追溯。

**这不是开箱即用的硬件部署包。** 源码快照仍有原工作区路径和跨目录依赖；运行硬件前必须在本机重新提供匹配的 SDK、权重、显示映射、ROI/倍率、标定和启动配置。不要直接运行历史 `switch.py` / `update.py` 等维护脚本。当前提交不改变实验台运行程序，也不代表边缘恢复、抗扰或实时性已经验收。

维护规则见 [数据与代码边界](docs/MAINTENANCE.md)。
