import jsonlines
import os
import json
import re
from tqdm import tqdm
from run_c import run_c_on_testcases
from run_rust import run_rust_on_testcases
import subprocess
from step_process import extract_steps_and_code_structured

def extract_code(infer_result):
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

def post_process(infer_result):
    match = re.search(r"```(?:\w+)?\n(.*?)```", infer_result, re.DOTALL)
    code = match.group(1).strip() if match else infer_result.strip()
    return code

def extract_testcases(raw_text):
    # 去掉 ```json 和 ```
    json_text = re.sub(r"```json|```", "", raw_text).strip()
    # 解析为 Python 对象
    json_text = json_text.replace("'", '"')
    return json.loads(json_text)

def loose_compare(a: str, b: str) -> int:
    if a is None or b is None:
        return 0
    tokens_a = a.split()
    tokens_b = b.split()
    return 1 if tokens_a == tokens_b else 0

def pass_all_tests(expected: list, actual: list) -> bool:
    for i in range(len(expected)):
        expected_output = expected[i]['stdout']
        actual_output = actual[i]['stdout']
        if loose_compare(expected_output, actual_output) == 0:
            if expected[i]['returncode'] != actual[i]['returncode']:
                continue
            return False
    return True

def match_c_rust():
#先根据idx匹配一下C代码和Rust代码
    filepath = '/data1/luofeng/RL/verl/conduct_data/Rust_code/Qwen2.5-Coder-32B-Instruct/infer_result.jsonl'
    data = list(jsonlines.open(filepath, mode='r'))
    data_new = []
    try:
        for item in data:
            new_item = {
                "idx": item['oai_response']['idx'],
                "c_code": item['source_data']['c_code'],
                "rust_code": post_process(item["oai_response"]['response'][0])
            }
            data_new.append(new_item)
    except Exception as e:
        print(f"Error processing item: {e}")

    with open("c_rust_new.jsonl", "w", encoding="utf-8") as f:
        for item in data_new:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def match_c_rust_testcase():
    # 再根据idx匹配一下C代码和Rust代码和测试用例
    filepath_testcase = '/data1/luofeng/RL/verl/conduct_data/testcase_v3/Qwen2.5-Coder-32B-Instruct/infer_result.jsonl'
    filepath_c_rust = '/data1/luofeng/RL/verl/conduct_data/c_rust_new.jsonl'
    data_testcase = list(jsonlines.open(filepath_testcase, mode='r'))   
    data_c_rust = list(jsonlines.open(filepath_c_rust, mode='r'))
    data_new = []
    for i in range(len(data_c_rust)):
        try:
            testcases = extract_testcases(
                data_testcase[i]['oai_response']['response'][0]
            )
        except json.JSONDecodeError as e:
            print(f"Skip idx {data_c_rust[i]['idx']} because JSON error: {e}")
            continue

        new_item = {
            "idx": data_c_rust[i]['idx'],
            "c_code": data_c_rust[i]['c_code'],
            "rust_code": data_c_rust[i]['rust_code'],
            "testcases": testcases
        }

        data_new.append(new_item)
    with open("c_rust_testcase_new.jsonl", "w", encoding="utf-8") as f:
        for item in data_new:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
def get_expected_outputs():
    # 再根据C和Testcases得到预期输出
    filepath_c_rust_testcase = '/data1/luofeng/RL/verl/conduct_data/c_rust_testcase_new.jsonl'
    data = list(jsonlines.open(filepath_c_rust_testcase, mode='r'))
    data_new = []
    count = 0
    with open("c_rust_testcase_output_new.jsonl", "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
            try:
                c_code = item['c_code']
                # print(item["oai_response"]['response'][0])
                testcases = item['testcases']
                rust_code = item['rust_code']
                outputs = run_c_on_testcases(c_code, testcases)
                item_new = {
                    "c_code": c_code,
                    "testcases": testcases,
                    "expected_outputs": outputs,
                    "rust_code": rust_code  
                }
                f.write(json.dumps(item_new, ensure_ascii=False) + "\n")
                data_new.append(item_new)

            except Exception as e:
                count += 1
                continue
        print(f"Total items: {len(data)}, Filtered items: {len(data_new)}, Errors: {count}")

def run_rust_and_compare():
    file_path = "/data1/luofeng/RL/verl/conduct_data/c_rust_testcase_output_new.jsonl"
    data = list(jsonlines.open(file_path, mode='r'))
    # idx = 5
    # c_code = data[idx]['c_code']
    # expected_outputs = data[idx]['expected_outputs']
    # rust_code = data[idx]['rust_code']
    # testcases = data[idx]['testcases']
    # print("C Code:\n", c_code)
    # print("Rust Code:\n", rust_code)
    # print("Testcases:\n", testcases)
    # print("Expected Outputs:\n", expected_outputs)
    # result = run_rust_on_testcases(rust_code, testcases)
    # print("Actual Outputs:\n", result)
    # print("Pass all tests:", pass_all_tests(expected_outputs, result))

    true_idx_list = []
    for i, item in tqdm(enumerate(data), total=len(data), desc="Evaluating Rust code"):
        expected_outputs = item['expected_outputs']
        rust_code = item['rust_code']
        testcases = item['testcases']
        try:
            result = run_rust_on_testcases(rust_code, testcases)
        except Exception as e:
            print(f"Error running Rust code at idx {i}: {e}")
            continue
        if pass_all_tests(expected_outputs, result):
            true_idx_list.append(i)
    print("Number of correct Rust code:", len(true_idx_list))
    with open("correct_rust_indices.json", "w") as f:
        json.dump(true_idx_list, f)

with open("correct_rust_indices_step.json", "r") as f:
    correct_indices = json.load(f)
filepath = "/data1/luofeng/RL/verl/conduct_data/step_v3/Qwen2.5-Coder-32B-Instruct/infer_result.jsonl"
data = list(jsonlines.open(filepath, mode='r'))
data_new = []
for idx in range(len(data)):
    if idx not in correct_indices:
        c_code = data[idx]['source_data']['c_code']
        expected_outputs = data[idx]['source_data']['expected_outputs']
        steps, rust_code = extract_steps_and_code_structured(data[idx]['oai_response']['response'][0])
        testcases = data[idx]['source_data']['testcases']
        item = {
            "index": idx,
            "c_code": c_code,
            "steps": steps,
            "rust_code": rust_code,
            "expected_outputs": expected_outputs,
            "testcases": testcases
        }
        data_new.append(item)
with open("my_data_step_bad.jsonl", "w", encoding="utf-8") as f:
    for item in data_new:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")