import sys
sys.path.append("/data1/luofeng/RL/verl")

import re
from codebleu import calc_codebleu
import math
from conduct_data.run_rust import run_rust_on_testcases
from typing import Any, Dict, Optional
import json
import jsonlines
from functools import lru_cache
import os
from openai import OpenAI
import pandas as pd

C2RUST_STEP_SCORE_PROMPT = """You are given a C program, a step-by-step reasoning process for translating it into Rust, and the final Rust program.
Assign ONE overall score between 0 and 1 (a real number) to the reasoning process, based on how well it contributes to producing a correct and complete Rust translation.
Do NOT evaluate individual steps.

Guidelines:
- Close to 1: accurate, complete, and strongly supportive
- Around 0.5: partially helpful but flawed
- Close to 0: incorrect or misleading

C program:
{c_code}

{steps_and_rust}

Output a decimal number between 0 and 1 with at most 6 decimal places.
Output ONLY the number. Do NOT output anything else.
"""

def sigmoid_scale_codebleu(score, mu=0.5, k=8):
    return 1 / (1 + math.exp(-k * (score - mu)))

def extract_solution(infer_result):
    match = re.search(r"```(?:\w+)?\n(.*?)```", infer_result, re.DOTALL)
    code = match.group(1).strip() if match else infer_result.strip()
    if code:
        return code
    return None

def extract_solution_steps(infer_result):
    if not infer_result:
            return None

    match = re.search(
        r"<code>\s*([\s\S]*?)\s*</code>",
        infer_result
    )
    if not match:
        return None
    code = match.group(1)
    # 去掉 ```rust / ``` / ```rs 包裹
    code = re.sub(r"```(?:rust|rs)?", "", code)
    code = code.replace("```", "")
    return code.strip()

def loose_compare(a: str, b: str) -> int:
    if a is None or b is None:
        return 0
    # 更宽松：去空格 + 忽略大小写
    return 1 if a.strip().lower() == b.strip().lower() else 0
# def loose_compare(a: str, b: str) -> int:
#     if a is None or b is None:
#         return 0
#     tokens_a = a.split()
#     tokens_b = b.split()
#     return 1 if tokens_a == tokens_b else 0

def pass_all_tests(expected: list, actual: list) -> bool:
    for i in range(len(expected)):
        expected_output = expected[i]['stdout']
        actual_output = actual[i]['stdout']
        if loose_compare(expected_output, actual_output) == 0:
            if expected[i]['returncode'] != actual[i]['returncode']:
                continue
            return False
    return True

def pass_ratio(expected: list, actual: list) -> float:
    if not expected:
        return 0.0

    passed = 0
    for i in range(len(expected)):
        expected_output = expected[i]['stdout']
        actual_output = actual[i]['stdout']
        if loose_compare(expected_output, actual_output):
            passed += 1

    return passed / len(expected)

# def pass_ratio(expected: list, actual: list) -> float:
#     passed = 0
#     for i in range(len(expected)):
#         expected_output = expected[i]['stdout']
#         actual_output = actual[i]['stdout']
#         if loose_compare(expected_output, actual_output) == 1:
#             passed += 1
#     return passed / len(expected) if expected else 0.0

def is_equiv(testcases, expected_output, rust_code: str):
    actual_output = run_rust_on_testcases(rust_code, testcases)
    return pass_all_tests(expected_output, actual_output)


TEST_DATA_PATH = "/data1/luofeng/RL/verl/data/my_data_processed.jsonl"
DEBUG_REWARD = os.getenv("VERL_REWARD_DEBUG", "0") == "1"
DEBUG_MAX_PRINT = 3
_print_count = 0


@lru_cache(maxsize=1)
def load_test_db():
    """
    每个 Python 进程只加载一次 JSONL，返回:
        { index(int): item(dict) }
    """
    db = {}
    with jsonlines.open(TEST_DATA_PATH, mode="r") as reader:
        for item in reader:
            idx = item.get("index", None)
            if idx is None:
                continue
            try:
                idx = int(idx)
            except Exception:
                continue
            db[idx] = item
    return db

STEP_DATA_PATH = "/data1/luofeng/RL/verl/conduct_data/my_data_step.jsonl"
@lru_cache(maxsize=1)
def load_step_db():
    """
    每个 Python 进程只加载一次 JSONL，返回:
        { index(int): item(dict) }
    """
    db = {}
    with jsonlines.open(STEP_DATA_PATH, mode="r") as reader:
        for item in reader:
            idx = item.get("index", None)
            if idx is None:
                continue
            try:
                idx = int(idx)
            except Exception:
                continue
            db[idx] = item
    return db

def compute_bleu_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Custom CodeBLEU reward for Rust code generation (without data-flow)."""
    extra_info = extra_info or {}
    index = extra_info.get("index", None)
    try:
        index = int(index)
    except Exception:
        return 0.0
    item = load_step_db().get(index)
    rust_code = item.get("rust_code", "")
    
    answer = extract_solution_steps(solution_str)

    # 只用三项，不用 dataflow（最后一个 weight=0 已经足够，但 CodeBLEU 仍会尝试解析）
    # 为避免 warning，直接用 no_lang=True，跳过 dataflow 解析。
    result = calc_codebleu(
        [rust_code], 
        [answer],
        lang="rust",
        weights=(0.33, 0.33, 0.33, 0),  # dataflow=0
        tokenizer=None
    )

    codebleu_score = result['codebleu']
    scaled_score = sigmoid_scale_codebleu(codebleu_score)
    return scaled_score

# def compute_execution_reward(
#     data_source,
#     solution_str,
#     ground_truth,
#     extra_info=None,
#     **kwargs,
# ):
#     extra_info = extra_info or {}

#     index = extra_info.get("index", None)

#     try:
#         index = int(index)
#     except Exception:
#         return -1.0

#     item = load_step_db().get(index)
#     if item is None:
#         return -1.0

#     testcases = item.get("testcases", None)
#     expected_outputs = item.get("expected_outputs", None)

#     answer = extract_solution_steps(solution_str)
#     if answer is None:
#         return -1.0

#     # ===== 执行代码 =====
#     result = run_rust_on_testcases(answer, testcases)

#     # ===== 1️⃣ 编译失败：强惩罚 =====
#     if not result.get("compile_success", False):
#         return -1.0

#     # ===== 2️⃣ 执行通过率 =====
#     score = pass_ratio(expected_outputs, result.get("test_results", []))
#     # score ∈ [0, 1]

#     # ===== 3️⃣ 编译成功基础奖励（关键！）=====
#     # 防止 compile 成功 ≈ 全错
#     base_reward = 0.2

#     # ===== 4️⃣ 长度惩罚（防止输出过短/投机）=====
#     # 目标长度大致 200 tokens，可根据你数据调
#     answer_len = len(answer.split())
#     length_factor = min(answer_len / 200.0, 1.0)  # ∈ (0,1]

#     # ===== 5️⃣ 最终 reward（平滑 + 可学习）=====
#     reward = base_reward + 0.8 * score   # ∈ [0.2, 1.0]
#     reward = reward * length_factor

#     # ===== 6️⃣ 防止异常 =====
#     if not isinstance(reward, float):
#         return -1.0

#     return reward

def compute_execution_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """
    verl 自定义 reward function 的单样本签名：
      - data_source: 当前样本的数据源名
      - solution_str: 模型生成的完整 response 字符串
      - ground_truth: 当前样本的 reward_model.ground_truth
      - extra_info: 当前样本的 extra_info 字典
      - **kwargs: 来自 custom_reward_function.reward_kwargs 的全局参数
    """
    extra_info = extra_info or {}

    global _print_count

    if DEBUG_REWARD and _print_count < DEBUG_MAX_PRINT:
        print("=" * 50)
        print("[reward debug]")
        print("data_source:", data_source)
        print("extra_info:", extra_info)
        print("solution_str[:200]:", solution_str[:200])
        _print_count += 1

    index = extra_info.get("index", None)

    try:
        index = int(index)
    except Exception:
        return 0.0

    item = load_step_db().get(index)
    testcases = item.get("testcases", None)
    expected_outputs = item.get("expected_outputs", None)

    answer = extract_solution_steps(solution_str)
    if answer is None:
        return 0.0

    # return 1.0 if passed else -1.0
    result = run_rust_on_testcases(answer, testcases)
    if not result["compile_success"]:
        return -1.0   # 编译错误强惩罚
    
    # 部分通过
    score = pass_ratio(expected_outputs, result["test_results"])

    #如果一个都没有通过，给予-1.0的奖励，表示虽然编译成功了，但完全不正确。
    if score == 0.0:
        score = -1.0

    return score


#==============================调用打分模型评分==============================
client = OpenAI(
    api_key="c2rust-12345",
    base_url="http://10.249.43.221:8080/v1",
)

SYSTEM_CONTEXT = "You are an expert in C-to-Rust translation."

def infer_one(prompt: str, model_name: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_CONTEXT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        top_p=1,
        max_tokens=2048,
    )

    return response.choices[0].message.content.strip()

def compute_score_with_model(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """
    使用模型评分的版本，供调试使用。
    """
    extra_info = extra_info or {}
    index = extra_info.get("index", None)
    try:
        index = int(index)
    except Exception:
        return 0.0
    item = load_step_db().get(index)
    c_code = item.get("c_code", "")

    steps_and_rust = solution_str

    prompt = C2RUST_STEP_SCORE_PROMPT.replace("{c_code}", c_code).replace("{steps_and_rust}", steps_and_rust)
    score_str = infer_one(prompt, "Qwen2.5-Coder-7B-Instruct-sft-score")
    
    try:
        score = float(score_str)
        return score
    except ValueError:
        return 0.0

#===============接下来是把excution_reward和score with model的分数合起来，用两个超参来调权重================
def compute_combined_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    
    execution_weight=0.5
    model_weight=0.5
    execution_reward = compute_execution_reward(
        data_source, solution_str, ground_truth, extra_info, **kwargs
    )
    model_score = compute_score_with_model(
        data_source, solution_str, ground_truth, extra_info, **kwargs
    )

    combined_score = execution_weight * execution_reward + model_weight * model_score

    if DEBUG_REWARD:
        print(f"Execution Reward: {execution_reward}, Model Score: {model_score}, Combined Score: {combined_score}")

    return combined_score

if __name__ == "__main__":
    index = 5
    item = load_step_db().get(index)
    c_code = item.get("c_code", "")
    print("C Code:")
    print(c_code)