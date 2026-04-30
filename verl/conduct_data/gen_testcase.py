import os
import torch
import jsonlines
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import argparse
import sys
import json
import random
# 确保 Python 能找到模块
sys.path.append(os.path.dirname(__file__))
C2RUST_TESTCASE_PROMPT = """You are an expert in C program analysis and test case generation.

Task:
Given a C program, generate 5 valid and diverse standard input test cases.

Requirements:
1. Each test case must be a complete stdin input for one execution.
2. Inputs must strictly follow the scanf format used in the program.
3. All test cases must be valid and executable.
4. The 5 test cases should be diverse and cover different branches and edge cases.
5. Carefully respect loop-based input counts. The number of integers in each test case must match the program's loop logic.
6. Do NOT include any explanation.

Output format:
Return a JSON array of 5 elements.
Each element is a list of integers representing the stdin input in order.

Example (one-shot):

C program:
#include <stdio.h>

int main()
{
    int n, m, x[103], y;
    scanf("%d", &n);
    for(int i=0; i<n; i++)
    {
       scanf("%d", &m);
       for(int j=1; j<=m; j++)scanf("%d", &x[j]);
       h:
       for(int j=1; j<m; j++)
       {
           for(int g=j+1; g<=m; g++)
           {
               if((j-x[j])==(g-x[g]))
               {
                   y=x[j];
                   x[j]=x[g];
                   x[g]=y;
                   goto h;
               }
           }
       }

       for(int j=1; j<=m; j++)printf("%d ", x[j]);
       printf("\\n");
    }
}

Output:
```json
[
  [1, 1, 42],
  [1, 5, 5, 4, 3, 2, 1],
  [1, 4, 1, 3, 4, 2],
  [2, 6, 1, 1, 1, 1, 1, 1, 6, 10, 20, 30, 40, 50, 60],
  [2, 8, 100, -1, 50, 0, 3, 2, 1, -5, 7, 0, 0, 0, 0, 0, 0]
]
```
Now generate test cases for the following C program:
{c_code}
"""
class Infer:

    def __init__(self, model_name, api_url=None, api_key=None, batch_size=20, reasoning=False):
        self.model_name = model_name
        if api_url and api_key:
            self.api_url = api_url
            self.api_key = api_key
            self.use_openai = True
        else:
            self.use_openai = False
            self.tensor_parallel_size = torch.cuda.device_count()
            self.load_model()
        self.reasoning = reasoning
        self.batch_size = batch_size
        self.USER_CONTEXT = C2RUST_TESTCASE_PROMPT


        self.SYSTEM_CONTEXT = (
            "You are an expert in C-to-Rust translation."
        )
        
    def load_model(self):
        if hasattr(self, "model") and hasattr(self, "tokenizer"):
            return
        model_path = os.path.join("/data/jfeng/models", self.model_name)
        if (
            "deepseekcoder" in self.model_name
            or "codellama" in self.model_name
            or "starcoder" in self.model_name
            or "Qwen" in self.model_name
        ):
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=15384, 
                swap_space=20,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            raise ValueError("Model not supported")
    
    def load_data(self):
        datasets = []
        filepath = '/data1/luofeng/RL/verl/conduct_data/successful.jsonl'
        data = list(jsonlines.open(filepath, mode='r'))
        for item in data:
            datasets.append(item)
        self.data = datasets
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(text)
        return len(tokens)
    

    def construct_prompt(self, item):
        c_code = item["c_code"]

        # prompt = self.USER_CONTEXT.format(
        #     c_code=code
        # )
        prompt = self.USER_CONTEXT.replace("{c_code}", c_code)
        # print(prompt)
        # input("Press Enter to continue...")
        return prompt
    
    def construct_prompts(self):
        if not hasattr(self, "data"):
            self.load_data()
        if self.use_openai:
            return [self.construct_prompt(item)
                    for item in self.data]
        prompts = []
        for item in self.data:
            inst = self.construct_prompt(item)
            if "starcoder" in self.model_name:
                prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": inst,
                    },

                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": self.SYSTEM_CONTEXT,
                        },
                        {
                            "role": "user",
                            "content": inst,
                        },

                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            # print(prompt)
            # input("Press Enter to continue...")
            prompts.append(prompt)
        return prompts
            
    def save_result(self, results):
        output_path = f"./testcase_v3/{self.model_name}"
        os.makedirs(output_path, exist_ok=True)
        # idx = 0
        # for result in results:
        #     file_path = os.path.join(output_path, f"{idx}.json")
        #     open(file_path, "w").write(f"{json.dumps(result, indent=4)}")
        #     idx += 1
        with jsonlines.open(os.path.join(output_path, "infer_result.jsonl"), mode='w') as writer:
            for result in results:
                writer.write(result)
        print("Inference completed. Results saved to {}".format(output_path))
    
    def infer_vllm(self, prompts):
        results = []
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size

        for idx in tqdm(range(0, len(prompts), self.batch_size), desc="Running vLLM inference", total=total_batches):
            batch_prompts = prompts[idx:idx + self.batch_size]
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=2048,
            )
            response = self.model.generate(batch_prompts, sampling_params)
            for i, res in enumerate(response):
                answer_content = []
                for j in range(1):
                    answer_content.append(res.outputs[j].text.strip())
                result={
                    "oai_response":{
                        "idx": idx + i,
                        "model": self.model_name,
                        "prompt": batch_prompts[i],
                        "response": answer_content
                    },
                    "source_data": self.data[idx + i]
                }
                # print(answer_content)
                # input("Press Enter to continue...")
                results.append(result)

        self.save_result(results)
        
    def infer_openai(self, prompts):
        results = []
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        def call_openai(idx, prompt):
            try:
                if self.reasoning:
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_CONTEXT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        top_p=1,
                    )
                    reasoning_content = response.choices[0].message.reasoning_content.strip()
                    content = response.choices[0].message.content.strip()
                    answer_content = []
                    answer_content.append(content)
                    result={
                        "oai_response":{
                            "idx": idx,
                            "model": self.model_name,
                            "prompt": prompt,
                            "response": answer_content,
                            "reasoning": reasoning_content
                        },
                        "source_data": self.data[idx]
                    }
                else:
                    # print(prompt)
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_CONTEXT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=2048,
                        top_p=1,
                    )
                    content = response.choices[0].message.content.strip()
                    # print(content)
                    # input("Press Enter to continue...")
                    answer_content = []
                    answer_content.append(content)
                    result={
                        "oai_response":{
                            "idx": idx,
                            "model": self.model_name,
                            "prompt": prompt,
                            "response": answer_content
                        },
                        "source_data": self.data[idx]
                    }
                return result
            except Exception as e:
                print(f"Error on prompt {idx}: {e}")
                return {
                    'idx': idx,
                    'prompt': prompt,
                    'response': f"Error: {e}"
                }

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(call_openai, idx, prompt) for idx, prompt in enumerate(prompts)]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running inference"):
                results.append(future.result())

        results.sort(key=lambda x: x['oai_response']['idx'])
        self.save_result(results)
        
    def infer(self):
        prompts = self.construct_prompts()
        if self.use_openai:
            self.infer_openai(prompts)
        else:
            self.infer_vllm(prompts)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with LLM')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--api_url', type=str, default=None, help='API URL for OpenAI')
    parser.add_argument('--api_key', type=str, default=None, help='API key for OpenAI')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')
    parser.add_argument('--reasoning', action='store_true', help='Enable reasoning')
    args = parser.parse_args()
    infer_model = Infer(model_name=args.model_name, api_url=args.api_url, api_key=args.api_key, batch_size=args.batch_size, reasoning=args.reasoning)
    infer_model.infer()
