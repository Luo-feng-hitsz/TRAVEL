model_paths=(
    # base
    # "./models/codegemma-7b-it"
    # "./models/deepseekcoder-6.7b"
    # "./models/Qwen2.5-Coder-3B-Instruct"
    # "./models/Qwen2.5-Coder-7B-Instruct"
    # "./models/deepseekcoder-33b"
    # "./models/Qwen3-Coder-30B-Instruct"

    # sft: c2rust-sft-no-step
    # "./checkpoints/c2rust-sft-no-step/codegemma-7B-it/global_step_1870"
    # "./checkpoints/c2rust-sft-no-step/qwen2.5-3B/global_step_1870"
    # "./checkpoints/c2rust-sft-no-step/qwen2.5-7B/global_step_1870"

    # sft: c2rust-sft-step
    # "./checkpoints/c2rust-sft-step/qwen2.5-7B/global_step_2805"
    # "./checkpoints/c2rust-sft-step/qwen2.5-3B/global_step_2805"
    # "./checkpoints/c2rust-sft-step/codegemma-7B-it/global_step_1870"
    # "./checkpoints/c2rust-sft-step/deepseekcoder-6.7B/global_step_2805"

    # As the RL model utilizes LoRA adapters, ensure the input path points to the base SFT model.
    "./checkpoints/c2rust-sft-step/qwen2.5-3B/global_step_2805"
    # "./checkpoints/c2rust-sft-step/qwen2.5-7B/global_step_2805"
    # "./checkpoints/c2rust-sft-step/codegemma-7B-it/global_step_1870"
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