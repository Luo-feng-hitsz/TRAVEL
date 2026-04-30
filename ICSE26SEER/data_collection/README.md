# Reproduce the Experiments

This document provides comprehensive instructions for reproducing the experiments presented in our paper.

> [!IMPORTANT]
> **General Requirements**
>
> We use VLLM to accelerate model inference. Note that VLLM may have environment conflicts with the training environment. We recommend using Anaconda to create a separate environment specifically for the data collection process.
>
> You can install the specific VLLM version used in this paper as follows:
>
> ```bash
> git clone https://github.com/MARIO-Math-Reasoning/vllm
> cd vllm
> pip install -e .
> ```
>
> Ensure you set `CUDA_VISIBLE_DEVICES` to specify the 1 or 2 GPUs you wish to use for the experiments.

## Reproducing the Data Collection Process

To generate the initial dataset using Monte Carlo Tree Search, execute the following command. This step creates synthetic samples based on our seed data and generates solution paths:

```bash
python solver_demo.py \
--custom_cfg configs/mcts_code.yaml \
--qaf ../data/my_data_processed.jsonl
```
