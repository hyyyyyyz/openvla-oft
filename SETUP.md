# 环境配置说明

## 设置 Conda 环境

```bash
# 创建并激活 conda 环境
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# 安装 PyTorch
# 请使用适合您机器的命令：https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# 克隆 openvla-oft 仓库并使用 pip install 安装以下载依赖
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# 安装 Flash Attention 2 用于训练（https://github.com/Dao-AILab/flash-attention）
#   =>> 如果遇到困难，可以先尝试执行 `pip cache remove flash_attn`
pip install packaging ninja
ninja --version; echo $?  # 验证 Ninja --> 应返回退出码 "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
