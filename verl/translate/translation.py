import os
import torch
import jsonlines
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
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

C2RUST_STEP_PROMPT = """Translate the following C program into an equivalent Rust program.

Requirements:
- The Rust program must compile as a standalone program with `main`.
- Read all input using `std::io::stdin().read_to_string(...)` and parse using `split_whitespace()`.
- Use <step> and </step> as delimiters between reasoning steps.
- Use <code> and </code> to wrap the final Rust code.

Here is an example:

C_code:
#include <stdio.h>

int main() {
    int n;
    scanf("%d", &n);

    int arr[1000];
    int sum = 0;
    int count = 0;

    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
        if (arr[i] % 2 == 0) {
            sum += arr[i];
            count++;
        }
    }

    printf("%d %d\\n", count, sum);
    return 0;
}

Answer:

<step> 
1. Understand the C program functionality. - The program first reads an integer `n`. - Then it reads `n` integers into an array. - While reading, it checks whether each number is even. - For every even number, it increments `count` and adds the value to `sum`. - Finally, it prints `count` and `sum`. 
</step> 
<step> 
2. Identify the required Rust input structure and program form. - The Rust program must compile as a standalone program with `main`. - All input must be read using `std::io::stdin().read_to_string(...)`. - All values must be parsed using `split_whitespace()`. - Therefore, each `scanf` call in the C program should be translated into sequential parsing from a whitespace iterator in Rust. 
</step> 
<step> 
3. Apply translation rules and reason about Rust equivalents. - C `int` is translated to Rust `i32` for this example. - The fixed-size C array `int arr[1000]` can be represented safely in Rust as `Vec<i32>`, because only the first `n` elements are actually used. - The `for` loop over indices can be translated into a Rust range loop `for _ in 0..n`. - The evenness test `arr[i] % 2 == 0` is preserved directly. - The output `printf("%d %d\n", count, sum)` becomes `println!("{} {}", count, sum);`. - No `unsafe` is required because all operations can be expressed with safe Rust constructs. 
</step> 
<step> 
4. Construct the final Rust program. - Define `main`. - Read all stdin into a string. - Create an iterator with `split_whitespace()`. - Parse `n`, then parse `n` integers one by one. - Store the values in a vector to reflect the array behavior. - Update `count` and `sum` when an even number is encountered. - Print the final results in the same format as the C program. 
</step>
<code> 
```rust 
use std::io::{self, Read};

fn main() {
let mut input = String::new();
io::stdin().read_to_string(&mut input).unwrap();
let mut iter = input.split_whitespace();
let n: usize = iter.next().unwrap().parse().unwrap();
let mut arr: Vec<i32> = Vec::with_capacity(n);
let mut sum: i32 = 0;
let mut count: i32 = 0;

for _ in 0..n {
    let x: i32 = iter.next().unwrap().parse().unwrap();
    arr.push(x);

    if x % 2 == 0 {
        sum += x;
        count += 1;
    }
}

println!("{} {}", count, sum);
}
```
</code>
Now it is your turn. Translate the following C program into Rust, following the same reasoning steps and output format.
C_code:
{c_code}

Answer:
"""

C2RUST_NO_STEP_PROMPT = """Translate the following C program into an equivalent Rust program.

Requirements:
- Preserve the exact algorithm and output behavior.
- The Rust program must compile as a standalone program with `main`.
- Read all input using `use std::io::{self, Read}; io::stdin().read_to_string(...)` and parse using `split_whitespace()`.

C program:
{c_code}
"""

C2RUST_BASE_PROMPT = """Here is code in C programming lanaguge. Translate the following code from C to Rust programming lanaguge. 
Do not output any extra description or tokens other than the translated code. 
{c_code}
"""



MODEL_KEYWORDS = [
    "qwen",
    "codegemma",
    "deepseek",
    "llama",
    "mistral",
    "baichuan",
    "chatglm",
    "gpt",
    "claude",
]

PROJECT_KEYWORDS = [
    "c2rust",
]

MODEL_KEYWORDS = [kw.lower() for kw in MODEL_KEYWORDS]
PROJECT_KEYWORDS = [kw.lower() for kw in PROJECT_KEYWORDS]

def extract_model_name(path: str) -> str:
    parts = path.strip("/").split("/")
    
    for part in parts:
        lower_part = part.lower()
        if any(kw in lower_part for kw in MODEL_KEYWORDS):
            return lower_part
    
    return None
def extract_project_name(path: str) -> str:
    parts = path.strip("/").split("/")
    
    for part in parts:
        lower_part = part.lower()
        if any(kw in lower_part for kw in PROJECT_KEYWORDS):
            return lower_part
    
    return "base"

class Infer:

    def __init__(self, model_path, api_url=None, api_key=None, batch_size=200, reasoning=False, lora_path=None):
        self.model_path = model_path
        self.lora_path = lora_path
        if lora_path:
            self.use_lora = True
        else:
            self.use_lora = False
        self.model_name = extract_model_name(self.model_path)
        # 如果use_lora为True，说明是rl，那么把project_name设置为c2rust-rl-step-test
        if self.use_lora:
            self.project_name = extract_model_name(self.lora_path)
        else:
            self.project_name = extract_project_name(self.model_path)

        if "grpo" in self.project_name or "sft-step" in self.project_name:
            self.USER_CONTEXT = C2RUST_STEP_PROMPT
            use_prompt = "C2RUST_STEP_PROMPT"   
        elif "base" in self.project_name:
            self.USER_CONTEXT = C2RUST_BASE_PROMPT
            use_prompt = "C2RUST_BASE_PROMPT"
        else:
            self.USER_CONTEXT = C2RUST_NO_STEP_PROMPT
            use_prompt = "C2RUST_NO_STEP_PROMPT"
        
        self.SYSTEM_CONTEXT = (
            "You are an expert in C-to-Rust translation."
        )
        
        print(f"Using model: {self.model_name}, project: {self.project_name}, use_lora: {self.use_lora}")
        print(f"Using prompt:\n{use_prompt}")
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
        
        
    def load_model(self):
        self.model = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=8192, 
            swap_space=20,
            enable_lora=self.use_lora,
            max_lora_rank=32,  
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.use_lora:
            self.lora_request = LoRARequest(
                lora_name="default",
                lora_int_id=1,
                lora_path=self.lora_path,
            )

    def load_data(self):
        datasets = []
        dataset_path = (
                f'YOUR_DATASET_PATH'
            )
        filepath = os.path.join(dataset_path, "YOUR_DATA.jsonl")
        data = list(jsonlines.open(filepath, mode='r'))
        for item in data:
            datasets.append(item)
        self.data = datasets       
    
    def count_tokens(self, text: str) -> int:
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(text)
        return len(tokens)
    
    def build_few_shot_prompt(self, retrieve, k):
        few_shot = ""
        for i in range(min(k, len(retrieve))):
            example = retrieve[i]
            if example['score'] < 100:
                return few_shot
            few_shot += f"C code:\n```\n{example['c_code']}\n```\n"
            few_shot += f"This is the translated Rust code:\n```\n{example['rust_code']}\n```\n\n"
        return few_shot

    def construct_prompt(self, item):
        c_code = item["source_code"]
        prompt = self.USER_CONTEXT.replace("{c_code}", c_code)

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
            if "codegemma" in self.model_name:
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
        output_path = f"./github_infer_result/{self.model_name}/{self.project_name}"
        os.makedirs(output_path, exist_ok=True)
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
            if self.use_lora:
                print(self.lora_request)
                response = self.model.generate(batch_prompts, sampling_params, lora_request=self.lora_request)
            else:
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
    parser.add_argument('--model_path', type=str, required=True, help='Model path for vLLM')
    parser.add_argument('--api_url', type=str, default=None, help='API URL for OpenAI')
    parser.add_argument('--api_key', type=str, default=None, help='API key for OpenAI')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for inference')
    parser.add_argument('--reasoning', action='store_true', help='Enable reasoning')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to lora adapter')
    args = parser.parse_args()
    infer_model = Infer(model_path=args.model_path, api_url=args.api_url, api_key=args.api_key, batch_size=args.batch_size, reasoning=args.reasoning, lora_path=args.lora_path)
    infer_model.infer()
