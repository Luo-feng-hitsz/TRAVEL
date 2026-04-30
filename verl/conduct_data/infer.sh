models=(
    "Qwen2.5-Coder-32B-Instruct"
)

export CUDA_VISIBLE_DEVICES=2,3
export VLLM_USE_V1=0

for model in "${models[@]}"; do
    echo "Running model: $model"
    python3 ./gen_testcase.py --model_name ${model} 
done

