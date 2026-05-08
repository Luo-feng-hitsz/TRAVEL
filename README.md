# TRAVEL Inference Guide

This README provides a minimal working guide on how to run inference using a **trained model**, as well as how to perform inference with **LoRA**.
## Installation

Install all required dependencies with:

Python == 3.10.0

GCC 9.4.0

Rustc 1.81.0

Driver Version: 570.133.20     

CUDA Version: 12.8

Linux

conda 23.1.0
```sh
pip install -r requirements.txt
```
## Resources
---
### Data and Models
- All the training data is located in the `./data` directory
- Trained models and lora adapters are available on [Hugging face](https://huggingface.co/datasets/Lauranne111/TRAVEL_data/tree/main).
- Evaluation dataset(xCodeEval): `./translate/groundtruth`， We extracted different C and Rust solutions to the same problem from the evaluation set of [ntunlp/xCodeEval](https://github.com/ntunlp/xCodeEval.git), and further filtered them to ensure each pair passes all test cases.
- Evaluation dataset(OS-Bench): The data are available on [Hugging face](https://huggingface.co/datasets/Lauranne111/TRAVEL_data/tree/main), we collect the data from [torvalds/linux](https://github.com/torvalds/linux)
- Evaluation dataset(HW-Bench): Not yet public

---
## Inference
### 1. Inference Script Location

The inference script is located at:

```bash
/TRAVEL/verl/translate/infer.sh
```

### 2. Parameter setup

Typically, you need to specify the following parameters in the script:

model_paths: path to the used model
lora_paths: path to the LoRA adapter 
(If you are not using the post-RL model, simply remove the --lora_path argument from the script.)

---
## Evaluation
Dependency and other information please refer to [ntunlp/xCodeEval](https://github.com/ntunlp/xCodeEval.git)
1. Run the following command to use the docker
```
./translate/manage_docker.sh start
```
2. Modify the path in `./my_eval` and run:
```
python my_eval.py
```
## Training
使用
## References
- https://github.com/shuzhenggao/ICSE26SEER
- https://github.com/verl-project/verl