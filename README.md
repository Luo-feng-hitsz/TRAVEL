# TRAVEL 推理说明（简版）

这个 README 先给一个最小可用说明：如何用**训好的模型**，以及如何通过 **LoRA** 做推理。

## 1. 推理脚本位置

推理脚本在：

```bash
/TRAVEL/verl/translate/infer.sh
```

如果你在仓库根目录（`/TRAVEL`）下，可以直接执行：

```bash
bash verl/translate/infer.sh
```

---

## 2. 使用训好的模型进行推理（示例）

通常你需要在脚本里或命令行里指定：

- `MODEL_PATH`：训好的基础模型路径
- `INPUT_FILE`：待翻译/待推理输入文件
- `OUTPUT_FILE`：推理结果输出文件

一个示例（按你自己的路径改）：

```bash
MODEL_PATH=/path/to/your_trained_model \
INPUT_FILE=/path/to/input.jsonl \
OUTPUT_FILE=/path/to/pred.jsonl \
bash verl/translate/infer.sh
```

---

## 3. 使用 LoRA 进行推理（示例）

如果你希望在基础模型上加载 LoRA，通常需要再提供：

- `LORA_PATH`：LoRA adapter 路径

示例：

```bash
MODEL_PATH=/path/to/base_model \
LORA_PATH=/path/to/lora_adapter \
INPUT_FILE=/path/to/input.jsonl \
OUTPUT_FILE=/path/to/pred_lora.jsonl \
bash verl/translate/infer.sh
```

---

## 4. 常见说明

- 如果脚本内部已经写死了参数，直接改 `infer.sh` 里的路径即可。
- 如果你使用多卡或指定 GPU，可以在前面加：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash verl/translate/infer.sh
```

- 首次跑不通时，优先检查：模型路径、LoRA 路径、输入文件格式是否正确。

---

## 5. 后续可以补充

后面可以再补：

- 输入输出样例格式（json/jsonl）
- 批量推理参数说明
- 多语言翻译任务示例
- 常见报错排查
