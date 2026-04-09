# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

**项目主页: https://openvla-oft.github.io/**

**论文: https://arxiv.org/abs/2502.19645**

**演示视频: https://youtu.be/T3Zkkr_NTSA**

## 系统要求

推理：
* 1 张 GPU，约 16 GB 显存（适用于 LIBERO 仿真基准任务）
* 1 张 GPU，约 18 GB 显存（适用于 ALOHA 机器人任务）

训练：
* 1-8 张 GPU，27-80 GB 显存，取决于训练配置（默认使用 bfloat16 数据类型）。详细信息请参阅[项目主页的 FAQ](https://openvla-oft.github.io/#train-compute)。

## 快速开始

首先，设置 conda 环境（请参阅 [SETUP.md](SETUP.md) 中的说明）。

然后，运行以下 Python 脚本下载预训练的 OpenVLA-OFT 检查点并进行推理，生成动作块：

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# 实例化配置（请参阅 experiments/robot/libero/run_libero_eval.py 中 GenerateConfig 类的定义）
cfg = GenerateConfig(
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# 加载 OpenVLA-OFT 策略和输入处理器
vla = get_vla(cfg)
processor = get_processor(cfg)

# 加载 MLP 动作头以通过 L1 回归生成连续动作
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# 加载本体感投影器，将本体感映射到语言嵌入空间
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# 加载样本观测：
#   observation (dict): {
#     "full_image": 主要第三人称图像,
#     "wrist_image": 腕部相机图像,
#     "state": 机器人本体感知状态,
#     "task_description": 任务描述,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# 生成机器人动作块（未来动作序列）
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("生成的动作块：")
for act in actions:
    print(act)
```

## 安装

有关设置 conda 环境的说明，请参阅 [SETUP.md](SETUP.md)。

## 训练与评估

有关在 LIBERO 仿真基准任务套件上进行微调/评估的说明，请参阅 [LIBERO.md](LIBERO.md)。

有关在真实世界的 ALOHA 机器人任务上进行微调/评估的说明，请参阅 [ALOHA.md](ALOHA.md)。
