# OpenVLA-OFT 在 LIBERO 仿真基准上的应用

## 相关文件

评估
* `experiments/robot/libero/`：LIBERO 评估文件
  * `run_libero_eval.py`：LIBERO 评估脚本
  * `libero_utils.py`：LIBERO 评估工具
* `experiments/robot/`：通用评估工具文件
  * `openvla_utils.py`：OpenVLA 专用评估工具
  * `robot_utils.py`：其他评估工具

训练
* `vla-scripts/finetune.py`：VLA 微调脚本


## 环境配置

设置 conda 环境（请参阅 [SETUP.md](SETUP.md) 中的说明）。

克隆并安装 [LIBERO 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 及所需依赖包：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # 从 openvla-oft 根目录执行
```

（可选，如果您计划启动训练）要下载我们在微调实验中使用的 [LIBERO 数据集](https://huggingface.co/datasets/openvla/modified_libero_rlds)，请运行以下命令。这将下载 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 和 LIBERO-10 数据集（RLDS 格式，共约 10 GB）。您可以使用这些数据集来微调 OpenVLA 或训练其他方法。由于我们在下方提供了预训练的 OpenVLA-OFT 检查点，此步骤为可选。
请注意，这些数据集与原始 OpenVLA 项目中使用的数据集相同。如有需要，请参阅[此处](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup)了解如何下载原始非 RLDS 数据集的详细信息。
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

## 启动 LIBERO 评估

我们通过 LoRA（r=32）结合 OFT 配方对 OpenVLA 进行了微调，在四个 LIBERO 任务套件上进行：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 和 LIBERO-10（也称为 LIBERO-Long）。
在我们论文的初始版本中，我们为每个 LIBERO 任务套件独立训练了一个检查点。在论文的更新版本中，我们进行了一个额外的实验，训练了一个在所有四个任务套件组合数据上训练的单一策略（结果见附录的"额外实验"部分）。总体而言，任务特定策略和组合策略的结果相当：四个套件的平均成功率分别为 97.1% 和 96.8%。

以下是四个独立训练的 LIBERO OpenVLA-OFT 检查点：
* [moojink/openvla-7b-oft-finetuned-libero-spatial](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial)
* [moojink/openvla-7b-oft-finetuned-libero-object](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object)
* [moojink/openvla-7b-oft-finetuned-libero-goal](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal)
* [moojink/openvla-7b-oft-finetuned-libero-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-10)

以下是在所有四个任务套件组合数据上训练的 OpenVLA-OFT 检查点：
* [moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10)

要使用独立训练的检查点之一启动评估，请运行以下命令之一。每个命令都会自动下载上面列出的相应检查点。您可以设置 `TRANSFORMERS_CACHE` 和 `HF_HOME` 环境变量来更改检查点文件的缓存位置。

```bash
# 启动 LIBERO-Spatial 评估
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial

# 启动 LIBERO-Object 评估
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object

# 启动 LIBERO-Goal 评估
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal

# 启动 LIBERO-10（LIBERO-Long）评估
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10
```

要评估在所有四个任务套件组合数据上训练的策略，只需将上述命令中的 `--pretrained_checkpoint` 替换为 `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` 即可。

注意事项：
* 评估脚本默认运行 500 次试验（10 个任务 × 每次 50 回合）。您可以通过设置 `--num_trials_per_task` 来修改每个任务的试验次数。也可以通过 `--seed` 更改随机种子。脚本中还有其他参数；我们将其设置为与上述 OpenVLA-OFT 检查点配合使用的默认值。
* **注意：设置 `--center_crop True` 非常重要**，因为我们使用随机裁剪增强对 OpenVLA 进行了微调（我们在每个训练样本中取了 90% 面积的随机裁剪，因此在测试时我们只需要取中心 90% 的裁剪）。
* 评估脚本在本地记录结果。您也可以通过设置 `--use_wandb True` 并指定 `--wandb_project <PROJECT>` 和 `--wandb_entity <ENTITY>` 来在 Weights & Biases 中记录结果。
* 我们论文中报告的结果是使用 **Python 3.10.14、PyTorch 2.2.0 和我们的[自定义 transformers v4.40.1 分支](https://github.com/moojink/transformers-openvla-oft.git)** 在 **NVIDIA A100 GPU** 上获得的，取三个随机种子的平均值。请尽可能使用这些软件包版本。请注意，如果您使用的 GPU 不是 A100，结果可能会略有不同。如果差异较大，请提交 GitHub Issue，我们会进行调查。

## 在 LIBERO 数据集上微调

首先，如上 Setup 部分所述，下载 LIBERO 数据集：`libero_spatial_no_noops`、`libero_object_no_noops`、`libero_goal_no_noops`、`libero_10_no_noops`。（`"_no_noops"` 表示无空操作动作，即过滤掉了动作接近零的训练样本）。

然后，使用下面的 OFT 配置启动微调脚本，将第一行中的 `X` 替换为 GPU 数量。下面的命令使用我们在论文中使用的超参数在 LIBERO-Spatial 上启动微调。在此配置下，每 GPU 批量大小为 8 需要约 62 GB 显存，每 GPU 批量大小为 1 需要约 25 GB 显存。

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```

如果 `X = 8` 并评估 150K 步的检查点，上述训练命令应该能复现我们的 OpenVLA-OFT 结果。

您可以将 `libero_spatial_no_noops` 替换为 `libero_object_no_noops`、`libero_goal_no_noops` 或 `libero_10_no_noops`。您也可以修改其他参数——例如，如果您只想使用第三人称相机的单个输入图像并禁用本体感知状态输入，可以设置 `--num_images_in_input 1` 和 `--use_proprio False`。

一般来说，我们建议微调直到训练 L1 损失降到 0.01 以下并开始趋于平稳（使用上述配置，在 100K 步后学习率下降 10 倍的情况下，LIBERO-Spatial 在 150K 梯度步后应达到约 0.006 的 L1 损失）。但是，对于 LIBERO-Goal，我们发现 50K 步的检查点（在约 0.02 L1 损失时）表现最佳，原因不明。对于其他所有任务套件，我们发现 150K 步的检查点表现最佳。

请务必使用与训练时相同的设备/GPU 来测试您的策略！否则性能可能会大幅下降。如果您在与训练不同的 GPU 上进行测试（例如如果您在 H100 上训练，然后在 A100 上合并后测试），将 LoRA 权重合并到基础模型中可能会避免性能下降。您可以参考我们的脚本 [vla-scripts/merge_lora_weights_and_save.py](vla-scripts/merge_lora_weights_and_save.py) 来离线将 LoRA 适配器合并到基础模型中。如果您在微调期间已经将 LoRA 权重合并到基础 OpenVLA 模型中也没关系；您可以随时重新下载基础模型并重新合并，只要您仍有 LoRA 适配器（`merge_lora_weights_and_save.py` 会为您处理）。
