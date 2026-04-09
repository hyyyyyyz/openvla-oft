# OpenVLA-OFT+ 在真实世界 ALOHA 机器人任务中的应用

## 相关文件

评估
* `experiments/robot/aloha/`：ALOHA 训练和评估文件
  * `run_aloha_eval.py`：ALOHA 评估脚本（客户端侧；见下面的"服务端侧"）
  * `aloha_utils.py`：ALOHA 评估工具
  * 从原始 [ALOHA GitHub 仓库](https://github.com/tonyzhaozh/aloha) 复制的其他 ALOHA 机器人环境文件：
    * `constants.py`
    * `real_env.py`
    * `robot_utils.py`
* `experiments/robot/`：通用评估工具文件
  * `openvla_utils.py`：OpenVLA 专用评估工具
  * `robot_utils.py`：其他评估工具
* `vla-scripts/deploy.py`：VLA 服务端部署脚本（服务端侧）

注意：与 LIBERO 评估设置不同，我们在这里使用服务端-客户端接口。如果控制机器人的用户机器无法访问具有足够规格的本地 GPU 来运行微调的 VLA 策略，这将特别有用。

训练
* `experiments/robot/aloha/`：ALOHA 训练和评估文件
  * `preprocess_split_aloha_data.py`：ALOHA 数据预处理脚本
* `vla-scripts/finetune.py`：VLA 微调脚本

## 环境配置

设置用于训练策略和部署到 VLA 服务端的 conda 环境（请参阅 [SETUP.md](SETUP.md) 中的说明）。

## 在 ALOHA 机器人数据上微调

我们假设您已经在 ALOHA 机器人上收集了一组专家演示数据。

首先，使用我们的 `preprocess_split_aloha_data.py` 脚本预处理原始 ALOHA 数据集：将图像从 480x640 缩小到 256x256，并拆分为训练集和验证集。以下是我们论文中 `put X into pot` 任务的示例（每个回合有 3 种可能的目标物体）：

```bash
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /scr/moojink/data/aloha1_raw/put_green_pepper_into_pot/ \
  --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
  --percent_val 0.05
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /scr/moojink/data/aloha1_raw/put_red_pepper_into_pot/ \
  --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
  --percent_val 0.05
python experiments/robot/aloha/preprocess_split_aloha_data.py \
  --dataset_path /scr/moojink/data/aloha1_raw/put_yellow_corn_into_pot/ \
  --out_base_dir /scr/moojink/data/aloha1_preprocessed/ \
  --percent_val 0.05
```

然后，将预处理后的 ALOHA 数据集转换为与 OpenVLA 微调兼容的单一 RLDS 数据集。此过程与原始 OpenVLA 仓库相同。转换说明请参阅[此处](https://github.com/moojink/rlds_dataset_builder)（可用的 ALOHA 预处理转 RLDS 转换脚本示例在[此处](https://github.com/moojink/rlds_dataset_builder/blob/main/aloha1_put_X_into_pot_300_demos/aloha1_put_X_into_pot_300_demos_dataset_builder.py)；该脚本将上述三个预处理数据集转换为一个统一的 RLDS 数据集，包含训练/验证拆分）。

转换为 RLDS 后，通过在 `configs.py`（[此处](prismatic/vla/datasets/rlds/oxe/configs.py#L680)）、`transforms.py`（[此处](prismatic/vla/datasets/rlds/oxe/transforms.py#L928)）和 `mixtures.py`（[此处](prismatic/vla/datasets/rlds/oxe/mixtures.py#L216)）中添加条目，将数据集（对于上述示例任务，名称为 `aloha1_put_X_into_pot_300_demos`）注册到我们的数据加载器中。作为参考，在这些文件中都有我们论文中使用的 ALOHA 数据集的示例条目。

在微调之前，在 [`prismatic/vla/constants.py`](prismatic/vla/constants.py) 中设置所需的 ALOHA 动作块大小（请参阅 `ALOHA_CONSTANTS` 中的 `NUM_ACTIONS_CHUNK`）。我们默认将其设置为 25，因为我们，在 ALOHA 设置中使用了 25 Hz 的控制频率来降低存储成本和训练时间（同时保持机器人运动的平滑性）。如果您使用 50 Hz，建议将 `NUM_ACTIONS_CHUNK` 设置为 `50`。一般来说，1 秒长的动作块是一个好的默认值。请勿修改 `ACTION_PROPRIO_NORMALIZATION_TYPE`：由于 ALOHA 机器人动作空间是绝对关节角度，我们不想使用会裁剪异常值的归一化方案（如我们在 LIBERO 相对末端执行器姿态动作上使用的 Q1-Q99 归一化），因为这将阻止模型输出对解决任务至关重要的某些机器人关节角度。

现在开始微调！以下是使用我们的 OFT+ 配方在上述 `put X into pot` 任务上微调 OpenVLA 的示例命令（"OFT+" 中的"+"表示包含 FiLM 以增强语言对齐）。将第一行中的 `X` 替换为您可用的 GPU 数量。

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name aloha1_put_X_into_pot_300_demos \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film
```

如果 `X = 8` 并评估 100K 步的检查点，上述训练命令应该能复现我们在 `put X into pot` 任务上的 OpenVLA-OFT+ 结果。它将使用 3 个输入图像（1 个第三人称图像 + 2 个腕部相机图像）微调 OpenVLA。请注意，我们在某个时间点（上述命令中为 50K 步）使用学习率衰减，因为这样做可以加快训练收敛速度（从我们的经验来看，训练 L1 损失会急剧下降）。

微调最佳实践：
* 一般来说，我们建议微调直到训练 L1 损失降到 0.01 以下并开始趋于平稳。
  * 实现这一点的一种方法是使用我们默认的学习率 `5e-4` 进行微调，直到损失开始非常缓慢地下降，然后将学习率衰减 10 倍到 `5e-5`（这将使损失急剧下降），并继续训练直到训练 L1 损失最终趋于平稳。
* 根据您的数据集大小，您可能需要调整一些超参数。例如，如果您使用包含超过 300 个演示的大型数据集，您可能需要较晚衰减学习率并训练更长时间以获得最佳性能。过早衰减可能导致次优策略。
* 如果您的任务不需要良好的语言对齐（例如，如果只有一个语言指令），则不需要 FiLM；考虑设置 `--use_film False` 以训练更少的模型参数。
* 请务必使用与训练时相同的设备/GPU 来测试您的策略！否则性能可能会大幅下降。如果您在与训练不同的 GPU 上进行测试（例如如果您在 H100 上训练，然后在 A100 上合并后测试），将 LoRA 权重合并到基础模型中可能会避免性能下降。您可以参考我们的脚本 [vla-scripts/merge_lora_weights_and_save.py](vla-scripts/merge_lora_weights_and_save.py) 来离线将 LoRA 适配器合并到基础模型中。如果您在微调期间已经将 LoRA 权重合并到基础 OpenVLA 模型中也没关系；您可以随时重新下载基础模型并重新合并，只要您仍有 LoRA 适配器（`merge_lora_weights_and_save.py` 会为您处理）。

如果遇到任何问题，请提交新的 GitHub Issue。

## 启动 ALOHA 机器人评估

在您将用于启动 VLA 服务端的主 conda 环境（`openvla-oft`）中，安装服务端-客户端接口所需的几个包：

```bash
conda activate openvla-oft
pip install uvicorn fastapi json-numpy
```

在您将用于控制机器人的机器上，设置第二个 conda 环境，该环境将用于运行机器人环境、查询 VLA 服务端并在环境中执行动作：

```bash
# 创建并激活客户端 conda 环境
conda create -n openvla-oft-aloha python=3.10 -y
conda activate openvla-oft-aloha

# 安装 PyTorch
# 请使用适合您机器的命令：https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# 克隆 openvla-oft 仓库并使用 pip install 安装以下载依赖
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# 安装 ALOHA 机器人环境所需的包
pip install -r experiments/robot/aloha/requirements_aloha.txt
```

在用于运行模型推理的 GPU 机器上启动 VLA 服务端（使用 `openvla-oft` conda 环境）。以下是示例命令（请根据需要更改）：

```bash
python vla-scripts/deploy.py \
  --pretrained_checkpoint /PATH/TO/FINETUNED/MODEL/CHECKPOINT/DIR/ \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key aloha1_put_X_into_pot_300_demos
```

然后，运行 ALOHA 评估脚本。在 `vla_server_url` 参数中指定 VLA 服务端 URL 或 IP 地址。以下是示例命令：

```bash
python experiments/robot/aloha/run_aloha_eval.py \
  --center_crop True \
  --num_open_loop_steps 25 \
  --use_vla_server True \
  --vla_server_url <VLA 服务端 URL> \
  --num_rollouts_planned <测试回合数> \
  --max_steps <每个回合最大步数>
```

如果遇到任何问题，请提交新的 GitHub Issue。

## 故障排除提示

* 提示 #1：如果遇到 ROS 错误，例如 `ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`，请在您的客户端 conda 环境（`openvla-oft-aloha`）中运行以下命令：

    ```
    conda install -c conda-forge libffi
    ```
