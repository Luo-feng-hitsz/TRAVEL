import subprocess
import tempfile
import os
import jsonlines
import json
import re
from tqdm import tqdm
# from step_process import extract_steps_and_code_structured

def run_rust_on_testcases(rust_code: str, inputs: list, timeout: float = 2.0):
    """
    timeout: 每个测试用例最大运行时间（秒）
    """
    outputs = []

    import tempfile, subprocess, os

    # 1️⃣ 写入临时 Rust 文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".rs") as f:
        rust_path = f.name
        f.write(rust_code.encode())

    exe_path = rust_path[:-3]

    try:
        # 2️⃣ 编译 Rust
        compile_result = subprocess.run(
            ["rustc", rust_path, "-O", "-o", exe_path],
            capture_output=True,
            text=True
        )

        # ❗关键修改：不再 raise，而是返回结构化信息
        if compile_result.returncode != 0:
            return {
                "compile_success": False,
                "compile_error": compile_result.stderr,
                "test_results": []
            }

        # ✅ 编译成功
        for value in inputs:
            all_input = "\n".join(map(str, value)) + "\n"

            try:
                result = subprocess.run(
                    [exe_path],
                    input=all_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                outputs.append({
                    "status": "OK",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                })

            except subprocess.TimeoutExpired as e:
                outputs.append({
                    "status": "TIMEOUT",
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                    "returncode": None
                })

        return {
            "compile_success": True,
            "compile_error": "",
            "test_results": outputs
        }

    finally:
        # 4️⃣ 清理文件
        if os.path.exists(rust_path):
            os.remove(rust_path)
        if os.path.exists(exe_path):
            os.remove(exe_path)

if __name__ == "__main__":
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

    file_path = "/data1/luofeng/RL/verl/conduct_data/step_v3/Qwen2.5-Coder-32B-Instruct/infer_result.jsonl"
    data = list(jsonlines.open(file_path, mode='r'))
    # idx = 358
    # c_code = data[idx]['source_data']['c_code']
    # expected_outputs = data[idx]['source_data']['expected_outputs']
    # steps, rust_code = extract_steps_and_code_structured(data[idx]['oai_response']['response'][0])
    # testcases = data[idx]['source_data']['testcases']
    # print("C Code:\n", c_code)
    # print("Steps:\n", steps)
    # print("Rust Code:\n", rust_code)
    # print("Testcases:\n", testcases)
    # print("Expected Outputs:\n", expected_outputs)
    # result = run_rust_on_testcases(rust_code, testcases)
    # print("Actual Outputs:\n", result)
    # print("Pass all tests:", pass_all_tests(expected_outputs, result))

    true_idx_list = []
    for i, item in tqdm(enumerate(data), total=len(data), desc="Evaluating Rust code"):
        expected_outputs = item['source_data']['expected_outputs']
        steps, rust_code = extract_steps_and_code_structured(item['oai_response']['response'][0])
        testcases = item['source_data']['testcases']
        try:
            result = run_rust_on_testcases(rust_code, testcases)
        except Exception as e:
            print(f"Error running Rust code at idx {i}: {e}")
            continue
        if pass_all_tests(expected_outputs, result):
            true_idx_list.append(i)
    print("Number of correct Rust code:", len(true_idx_list))
    with open("correct_rust_indices_step.json", "w") as f:
        json.dump(true_idx_list, f)