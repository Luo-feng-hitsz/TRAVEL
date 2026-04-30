model_paths=(
    # base模型
    # "./models/codegemma-7b-it"
    # "./models/deepseekcoder-6.7b"
    # "./models/Qwen2.5-Coder-3B-Instruct"
    # "./models/Qwen2.5-Coder-7B-Instruct"
    # "./models/deepseekcoder-33b"
    # "./models/Qwen3-Coder-30B-Instruct"

    # sft模型: c2rust-sft-original
    # "./checkpoints/c2rust-sft-original/qwen2.5-3B/global_step_1686"
    # "./checkpoints/c2rust-sft-original/codegemma-7B-it/global_step_1686"
    # "./checkpoints/c2rust-sft-original/qwen2.5-7B/global_step_1686"

    # sft模型: c2rust-sft-no-step-v3（和v2一样的数据，用的更简单的prompt）
    # "./checkpoints/c2rust-sft-no-step-v3/codegemma-7B-it/global_step_1870"
    # "./checkpoints/c2rust-sft-no-step-v3/qwen2.5-3B/global_step_1870"
    # "./checkpoints/c2rust-sft-no-step-v3/qwen2.5-7B/global_step_1870"

    # sft模型: c2rust-sft-step-v3
    # "./checkpoints/c2rust-sft-step-v3/qwen2.5-7B/global_step_2805"
    # "./checkpoints/c2rust-sft-step-v3/qwen2.5-3B/global_step_2805"
    # "./checkpoints/c2rust-sft-step-v3/codegemma-7B-it/global_step_1870"
    # "./checkpoints/c2rust-sft-step-v3/deepseekcoder-6.7B/global_step_2805"

    # rl模型由于用的都是lora，所以这里传入的路径应该是对应的sft模型的路径
    "./checkpoints/c2rust-sft-step-v3/qwen2.5-3B/global_step_2805"
    # "./checkpoints/c2rust-sft-step-v3/qwen2.5-7B/global_step_2805"
    # "./checkpoints/c2rust-sft-step-v3/codegemma-7B-it/global_step_1870"
)

lora_paths=(
    "./lora_set/qwen2.5-3B-grpo/lora_adapter"
    # "./lora_set/qwen2.5-3B-grpo-no-stepreward/lora_adapter"
    # "./lora_set/qwen2.5-3B-grpo-no-testreward/lora_adapter"

    # "./lora_set/qwen2.5-7B-grpo/lora_adapter"
    # "./lora_set/qwen2.5-7B-grpo-no-stepreward/lora_adapter"
    # "./lora_set/qwen2.5-7B-grpo-no-testreward/lora_adapter"

    # "./lora_set/codegemma-7B-grpo/lora_adapter"
    # "./lora_set/codegemma-7B-grpo-no-stepreward/lora_adapter"
    # "./lora_set/codegemma-7B-grpo-no-testreward/lora_adapter"
)

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=0

for model_path in "${model_paths[@]}"; do
    echo "Running model: $model_path"
    python3 ./translation.py --model_path $model_path --lora_path ${lora_paths[0]}
    # python3 ./translation.py --model_path $model_path
done

# python3 ./translation.py --model_path "ANYTHING" --api_url "YOUR_URL" --api_key "YOUR_KEY"