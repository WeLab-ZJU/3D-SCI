# 3D-SCI



## 🧐 项目简介

**3D-SCI** 

**核心价值**：
- **鲁棒性**：利用扩散先验弥补局部特征缺失，即使重叠率低于 15% 也能稳定拼接。
- **灵活性**：支持 2D 切片序列或 3D 堆栈的直接输入，兼容显微镜、CT、MRI 等多种模态。
- **完整性**：提供从 `Train.py`（训练扩散先验）到 `Run.py`（一键推理）的全链路工具，并内置 `ImgEval` 模块自动量化拼接质量（PSNR、SSIM 等）。
- **效率**：基于 PyTorch 2.4 + CUDA 12.4 加速，可处理 GB 级数据。

---

## 📦 安装

### 环境要求
- Python 3.9（推荐，3.8-3.10 也可）
- CUDA 12.4（若使用 GPU，其他版本需调整 PyTorch 安装源）
- 至少 16GB 内存（推荐 32GB+，处理大体积数据）

### 1. 克隆仓库
```bash
git clone https://github.com/[YourUsername]/3D-SCI-public.git
cd 3D-SCI-public
