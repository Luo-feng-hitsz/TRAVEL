set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m verl.trainer.main_ppo \
  --config-path=./verl/trainer/config \
  --config-name=my_grpo_trainer 2>&1 | tee c2rust_rl_3b.log
