export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1
vllm serve /data1/luofeng/RL/checkpoints/c2rust-sft-step-score/qwen2.5-7B/global_step_711\
    --api-key c2rust-12345\
    --max-model-len 16384\
    --tensor-parallel-size 2\
    --served-model-name Qwen2.5-Coder-7B-Instruct-sft-score\
    --trust-remote-code\
    --port 8080
    # --enable-reasoning --reasoning-parser deepseek_r1\