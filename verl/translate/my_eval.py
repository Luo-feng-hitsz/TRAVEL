from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import jsonlines
import requests
import pandas as pd
from tqdm import tqdm
import re
import os
import json
from codebleu import calc_codebleu
from unsafe_detect import analyze_rust_unsafe_blocks
import warnings
warnings.filterwarnings("ignore")



class evaluate():
    _session: requests.Session
    
    def __init__(self, max_workers: int = 100,
                 server_url: str = "http://localhost:5000"):
        self.max_workers = max_workers
        self._session = requests.Session()
        self.execute_code_url = f"{server_url}/api/execute_code"
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()
    
    def load_data(self, dirpath):
        self.unittests = []
        self.src_uids = []
        self.rust_code = []
        self.ground_truth = []
        filepath = os.path.join(dirpath, "infer_result.jsonl")
        src_uids = json.load(open("/data/luofeng/c2rust/dataset/src_uids.json", "r", encoding="utf-8"))
        data = list(jsonlines.open(filepath, mode='r'))
        for item in data:
            if item['source_data']['src_uid'] not in src_uids:
                continue
            self.unittests.append(json.loads(item['source_data']['hidden_unit_tests']))
            self.src_uids.append(item['source_data']['src_uid'])
            # print(item['oai_response']['response'][0])
            # input("press enter...")
            self.rust_code.append(item['oai_response']['response'][0])
            self.ground_truth.append(item['source_data']['target_code'])
    
    def execute_code(
        self,
        source_code: str,
        unittests: List[dict],
        compiler: str = "Rust 2018",
        limits: Optional[dict] = None,
        block_network: bool = True,
        stop_on_first_fail: bool = True,
        use_sanitizer: bool = False,
        compiler_program_name: Optional[str] = None,
        compiler_flags: Optional[str] = None,
        interpreter_cmd: Optional[str] = None,
        interpreter_flags: Optional[str] = None,
        sample_id: Optional[int] = None,
        task_id: Union[str, int, None] = None,
    ):
        request_body = dict(
            language=compiler,
            source_code=source_code,
            unittests=unittests,
            limits=limits if isinstance(limits, dict) else None,
            compile_cmd=compiler_program_name,
            compile_flags=compiler_flags,
            execute_cmd=interpreter_cmd,
            execute_flags=interpreter_flags,
            block_network=block_network,
            stop_on_first_fail=stop_on_first_fail,
            use_sanitizer=use_sanitizer,
        )
        json_response = self._session.post(
            self.execute_code_url,
            json=request_body,
            headers={"Content-Type": "application/json"},
        ).json()

        if "data" not in json_response:
            return json_response, sample_id, task_id

        return (
            json_response["data"],
            sample_id,
            task_id,
        )
    
    def post_process(self, infer_result):
        match = re.search(r"```(?:\w+)?\n(.*?)```", infer_result, re.DOTALL)
        code = match.group(1).strip() if match else infer_result.strip()
        return code

    def extract_code(self, infer_result):
        if not infer_result:
            return None

        text = infer_result.strip()

        match = re.search(r"<code>\s*([\s\S]*?)\s*</code>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        match = re.search(r"```(?:rust|rs)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return text

    # def extract_code(self, infer_result):
    #     rust_match = re.search(
    #         r"<code>\s*(.*?)\s*</code>",
    #         infer_result,
    #         re.DOTALL
    #     )

    #     rust_code = None
    #     if rust_match:
    #         rust_block = rust_match.group(1)
    #         # 去掉 ```rust ``` 包裹
    #         rust_code = re.sub(r"```rust|```", "", rust_block).strip()

    #     return rust_code
    
    # def analyze_runtime_result(self, runtime_result):
    #     for result in runtime_result["runtime_result"]:
    #         if result.get('exec_outcome') == 'PASSED':
    #             continue
    #         return False
    #     return True
    def analyze_runtime_result(self, runtime_result):
        rr = runtime_result.get("runtime_result")

        # 情况 1：runtime_result 不是正常的 list（如 error dict / None）
        if not isinstance(rr, list):
            return False

        for result in rr:
            # 情况 2：单条结果结构异常
            if not isinstance(result, dict):
                return False

            # 如果没有 exec_outcome，说明执行结果异常（如模型生成 None）
            if "exec_outcome" not in result:
                return False

            # 只要有一个没通过，就整体失败
            if result.get("exec_outcome") != "PASSED":
                return False

        # 所有 case 都 PASSED
        return True

    def analyze_runtime_result_compile(self, runtime_result):
        rr = runtime_result.get("runtime_result")

        # 情况 1：runtime_result 结构异常（如 error dict / None）
        if not isinstance(rr, list):
            return False

        for result in rr:
            # 情况 2：单条结果结构异常
            if not isinstance(result, dict):
                return False

            # 没有 exec_outcome，通常意味着生成 / 执行阶段失败
            if "exec_outcome" not in result:
                return False

            # 只要出现编译错误，就判定失败
            if result.get("exec_outcome") == "COMPILATION_ERROR":
                return False

        # 未出现任何编译错误
        return True

    def evaluate(self, dirpath):
        result_dir = os.path.join(dirpath, "result")
        os.makedirs(result_dir, exist_ok=True)
        self.load_data(dirpath)

        tasks = []
        for infer_result, src_uid, unittest in zip(self.rust_code, self.src_uids, self.unittests):
            code = self.extract_code(infer_result)
            # print(code)
            # input("Press Enter to continue...")
            tasks.append((code, unittest, src_uid))

        runtime_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            features = {
                executor.submit(self.execute_code, code, unittest, task_id=src_uid): src_uid
                for code, unittest, src_uid in tasks
            }
            for future in tqdm(as_completed(features), total=len(features), desc="Evaluating"):
                src_uid = features[future]
                try:
                    runtime_result, _, _ = future.result()
                    result = {
                        "src_uid": src_uid,
                        "runtime_result": runtime_result
                    }
                    runtime_results.append(result)
                except Exception as e:
                    print(f"Error processing task {src_uid}: {e}")
                    runtime_results.append({
                        'exec_outcome': 'ERROR',
                        'error_message': str(e),
                    })
        # 计算CA
        if not runtime_results:
            score = 0.0
        else:
            # print(result)
            # input("Press Enter to continue...")
            score = sum(
                1 if self.analyze_runtime_result(result) else 0
                for result in runtime_results
            ) / len(runtime_results)

        # 计算EA
        if not runtime_results:
            ea_score = 0.0
        else:
            ea_score = sum(
                1 if self.analyze_runtime_result_compile(result) else 0
                for result in runtime_results
            ) / len(runtime_results)

        # # 计算codebleu分数
        # idx = 0
        # for infer_result in self.rust_code:
        #     if self.extract_code(infer_result) == '':
        #         score_df = pd.DataFrame({
        #             "Computational Accuracy": [f"{score * 100:.2f}%"],
        #             "Execution Accuracy": [f"{ea_score * 100:.2f}%"],
        #         })
        #         score_df.to_csv(f"{result_dir}/score.csv", index=False)
        #         print(f"Computational Accuracy: {score * 100:.2f}%")
        #         print(f"Execution Accuracy: {ea_score * 100:.2f}%")
        #         return
        # code = [self.extract_code(infer_result) for infer_result in self.rust_code]

        # # codebleu_score = calc_codebleu(code, self.ground_truth, lang="rust", weights=(0.33, 0.33, 0.33, 0), tokenizer=None)
        # codebleu_score = 0
        # for prediction, reference in zip(code, self.ground_truth):
        #     if prediction == None or prediction.strip() == "":
        #         continue
        #     result = calc_codebleu([reference], [prediction], lang="rust", weights=(0.33, 0.33, 0.33, 0), tokenizer=None)
        #     if result['dataflow_match_score'] != 0:
        #         result = calc_codebleu([reference], [prediction], lang="rust", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
        #     codebleu_score += result['codebleu']
        # codebleu_score /= len(self.ground_truth)

        # 计算unsafe rate 和 unsafe loc rate
        code = [self.extract_code(infer_result) for infer_result in self.rust_code]
        unsafe_rate = 0
        unsafe_loc_rate = 0
        for rust_code in code:
            if rust_code == None or rust_code.strip() == "":
                continue
            counts, percentages = analyze_rust_unsafe_blocks(rust_code)
            unsafe_loc_rate += percentages['all_unsafe']
            if percentages['all_unsafe'] != 0:
                unsafe_rate += 1
        unsafe_rate /= len(self.ground_truth)
        unsafe_loc_rate /= len(self.ground_truth)

        score_df = pd.DataFrame({
            "Computational Accuracy": [f"{score * 100:.2f}%"],
            "Execution Accuracy": [f"{ea_score * 100:.2f}%"],
            # "CodeBLEU": [f"{codebleu_score * 100:.2f}%"],
            "Unsafe Rate": [f"{unsafe_rate * 100:.2f}%"],
            "Unsafe Loc Rate": [f"{unsafe_loc_rate * 100:.2f}%"],
        })
        score_df.to_csv(f"{result_dir}/score.csv", index=False)
        print(f"Computational Accuracy: {score * 100:.2f}%")
        print(f"Execution Accuracy: {ea_score * 100:.2f}%")
        # print(f"CodeBLEU: {codebleu_score * 100:.2f}%")
        print(f"Unsafe Rate: {unsafe_rate * 100:.2f}%")
        print(f"Unsafe Loc Rate: {unsafe_loc_rate * 100:.2f}%")
        with jsonlines.open(f"{result_dir}/runtime_results.jsonl", mode='w') as writer:
            for result in runtime_results:
                writer.write(result)
        
if __name__ == "__main__":
    with evaluate() as evaluator:
        # root_dir = "./verl/translate/infer_result/codegemma-7B-it"
        # for tmp_dir in os.listdir(root_dir):
        #     if "starcoder" in tmp_dir:
        #         continue
        #     tmppath = os.path.join(root_dir, tmp_dir)
        #     print(f"Evaluating {tmppath}")
        #     evaluator.evaluate(tmppath)
        evaluator.evaluate("./verl/translate/infer_result/codegemma-7b-it/VERT_fixed")