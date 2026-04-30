import subprocess
import tempfile
import os
import jsonlines
import json
import re
from tqdm import tqdm

def run_rust_on_testcases(rust_code: str, inputs: list, timeout: float = 2.0):
    """
    timeout: 每个测试用例最大运行时间（秒）
    """
    outputs = []

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

        if compile_result.returncode != 0:
            raise RuntimeError("Rust 编译失败:\n" + compile_result.stderr)

        # 3️⃣ 逐个输入运行
        for value in inputs:
            all_input = "\n".join(map(str, value)) + "\n"
            # all_input = " ".join(map(str, value)) + "\n"
            # print(f"Running Rust code with input:\n{all_input}")
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

        return outputs

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
    def pass_ratio(expected: list, actual: list) -> float:
        passed = 0
        for i in range(len(expected)):
            expected_output = expected[i]['stdout']
            actual_output = actual[i]['stdout']
            if loose_compare(expected_output, actual_output) == 1:
                passed += 1
        return passed / len(expected) if expected else 0.0


    file_path = "/data1/luofeng/RL/verl/conduct_data/my_data.jsonl"
    data = list(jsonlines.open(file_path, mode='r'))
    idx = 3599
    c_code = data[idx]['c_code']
    expected_outputs = data[idx]['expected_outputs']
    rust_code = data[idx]['rust_code']
    testcases = data[idx]['testcases']
    print("C Code:\n", c_code)
    print("Rust Code:\n", rust_code)
    print("Testcases:\n", testcases)
    print("Expected Outputs:\n", expected_outputs)
    result = run_rust_on_testcases(rust_code, testcases)
    print("Actual Outputs:\n", result)
    print("Pass all tests:", pass_all_tests(expected_outputs, result))
    print("Pass ratio:", pass_ratio(expected_outputs, result))

    # true_idx_list = []
    # for i, item in tqdm(enumerate(data), total=len(data), desc="Evaluating Rust code"):
    #     expected_outputs = item['expected_outputs']
    #     rust_code = item['rust_code']
    #     testcases = item['testcases']
    #     try:
    #         result = run_rust_on_testcases(rust_code, testcases)
    #     except Exception as e:
    #         print(f"Error running Rust code at idx {i}: {e}")
    #         continue
    #     if pass_all_tests(expected_outputs, result):
    #         true_idx_list.append(i)
    # print("Number of correct Rust code:", len(true_idx_list))
    # with open("correct_rust_indices.json", "w") as f:
    #     json.dump(true_idx_list, f)

    #统计通过率分布情况，比如1.0有多少，0.6有多少，0.0有多少
    true_idx_list = [0] * 11  # 索引0-10对应0.0-1.0
    for i, item in tqdm(enumerate(data[:1000]), total=len(data[:1000]), desc="Evaluating Rust code"):
        expected_outputs = item['expected_outputs']
        rust_code = item['rust_code']
        testcases = item['testcases']
        try:
            result = run_rust_on_testcases(rust_code, testcases)
        except Exception as e:
            print(f"Error running Rust code at idx {i}: {e}")
            continue
        ratio = pass_ratio(expected_outputs, result)
        index = int(ratio * 10)
        true_idx_list[index] += 1
    print("Pass ratio distribution (0.0 to 1.0):")
    for i in range(11):
        print(f"{i/10:.1f}: {true_idx_list[i]}")
