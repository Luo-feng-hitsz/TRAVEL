set -x
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer --config-path ./verl/trainer/config --config-name my_sft_trainer.yaml 2>&1 | tee c2rust-sft-no-step.log

