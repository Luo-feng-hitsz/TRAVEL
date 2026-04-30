import pandas as pd
import jsonlines
import json
import os
import math

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

C2RUST_NO_STEP_PROMPT_NEW = """Here is code in C programming lanaguge. Translate the following code from C to Rust programming lanaguge. 
Do not output any extra description or tokens other than the translated code. 
{c_code}
"""
max_token_len = 8192
def normalize_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return x[0]
    if hasattr(x, "item"):  # numpy scalar / ndarray
        return x.item()
    raise TypeError(type(x))

def cal_average_score(steps) -> float:
    values = []
    for step in steps:
        score = step["step_score"]
        if isinstance(score, (int, float)):
            values.append(float(score))
        else:
            pass
    return sum(values) / len(values) if values else 0

def process_2_prompt(steps, rust_code):
#把steps, rust_code和rust_code拼接成一个字符串
    result = ""

    step_texts = []
    for step in steps:
        step_texts.append(step["content"].strip())
    # 如果 steps 是 list
    index = 1
    for step_text in step_texts:
        result += "<step>\n"
        result += f"{index}. {step_text.strip()}\n"
        result += "</step>\n"
        index += 1

    # 加上代码部分
    result += "<code>\n"
    result += rust_code.strip() + "\n"
    result += "</code>"

    return result
def process_2_parquet_sft_step_score(data_set, out_path: str):
    rows = []
    for obj in data_set:
        c_code = obj.get("c_code", "")
        rust_code = obj.get("rust_code", "")
        steps = obj.get("steps", [])
        steps_and_rust = process_2_prompt(steps, rust_code)
        prompt = C2RUST_STEP_SCORE_PROMPT.replace("{c_code}", c_code).replace("{steps_and_rust}", steps_and_rust)
        answer = cal_average_score(steps)
        # 把answer转为字符串，保留6位小数
        answer = f"{answer:.6f}"

        if len(prompt) > max_token_len:
            continue

        rows.append({
            "prompt": normalize_text(prompt),
            "answer": answer
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

def process_2_parquet_sft_step(data_set, out_path: str):
    rows = []
    for obj in data_set:
        c_code = obj.get("c_code", "")
        rust_code = obj.get("rust_code", "")
        steps = obj.get("steps", [])
        steps_and_rust = process_2_prompt(steps, rust_code)
        prompt = C2RUST_STEP_PROMPT.replace("{c_code}", c_code)
        answer = steps_and_rust

        if len(prompt) > max_token_len:
            continue

        rows.append({
            "prompt": normalize_text(prompt),
            "answer": answer
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

def process_2_parquet_rl_step_small(data_set, out_path: str):
    rows = []

    for item in data_set:
        c_code = item.get("c_code", "")
        prompt = C2RUST_STEP_PROMPT.replace("{c_code}", c_code)
        row = {
            "data_source": "c2rust",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": ""
            },
            "extra_info": {
                "index": int(item.get("index", -1))
            }
        }
        rows.append(row)
        # print("token length of prompt:", len(prompt))
        # input("enter...")
        # print(row)
        # input("enter...")
    df = pd.DataFrame(rows)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")

def process_2_parquet_sft_no_step(data_set, out_path: str):
    rows = []
    for obj in data_set:
        c_code = obj.get("c_code", "")
        rust_code = obj.get("rust_code", "")
        prompt = C2RUST_NO_STEP_PROMPT_NEW.replace("{c_code}", c_code)
        answer = rust_code

        if len(prompt) > max_token_len:
            continue
        
        # print("prompt:", prompt)
        # print("answer:", answer)
        # input("enter...")
        rows.append({
            "prompt": normalize_text(prompt),
            "answer": answer
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
if __name__ == "__main__":

# ===================================sft-step-score================================================
# # 读取jsonl文件，划分为训练集和测试集 
#     data_path = "/data1/luofeng/RL/verl/conduct_data/my_data_step_scored.jsonl"
#     data_posotive = list(jsonlines.open(data_path, mode='r'))
#     data_path = "/data1/luofeng/RL/verl/conduct_data/my_data_step_bad_scored.jsonl"
#     data_negative = list(jsonlines.open(data_path, mode='r'))
    
#     # 正负样本比例设置为4:1，划分训练集和测试集为9:1
#     data_posotive = data_posotive[:len(data_negative)*4]
#     train_num_pos = math.floor(len(data_posotive)*0.9)
#     train_num_neg = math.floor(len(data_negative)*0.9)
#     train_set = data_posotive[:train_num_pos] + data_negative[:train_num_neg]
#     test_set = data_posotive[train_num_pos:] + data_negative[train_num_neg:]
#     print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

#     # out_path = "/data1/luofeng/RL/verl/data/sft_data/train_step_score.parquet"
#     # process_2_parquet_sft_step_score(train_set, out_path)
#     # out_path = "/data1/luofeng/RL/verl/data/sft_data/test_step_score.parquet"
#     # process_2_parquet_sft_step_score(test_set, out_path)

#===================================step的数据但是没有加入step的sft==============================
    # 读取jsonl文件，划分训练集和测试集为9:1
    data_path = "/data1/luofeng/RL/verl/conduct_data/my_data_step.jsonl"
    data = list(jsonlines.open(data_path, mode='r'))
    print(f"Total samples: {len(data)}")

    train_num = math.floor(len(data)*0.9)
    train_set = data[:train_num]
    test_set = data[train_num:]
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    output_dir = os.path.join("/data1/luofeng/RL/verl/data/sft_data", "train_no_step.parquet")
    process_2_parquet_sft_no_step(train_set, output_dir)
    output_dir = os.path.join("/data1/luofeng/RL/verl/data/sft_data", "test_no_step.parquet")
    process_2_parquet_sft_no_step(test_set, output_dir)
# ===================================sft带有step的数据处理================================================
    # data_path = "/data1/luofeng/RL/verl/conduct_data/my_data_step.jsonl"
    # data = list(jsonlines.open(data_path, mode='r'))
    # print(f"Total samples: {len(data)}")
    # # 划分训练集和测试集为9:1
    # train_num = math.floor(len(data)*0.9)
    # train_set = data[:train_num]
    # test_set = data[train_num:]
    # print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    # out_path = "/data1/luofeng/RL/verl/data/sft_data/train_step.parquet"
    # process_2_parquet_sft_step(train_set, out_path)
    # out_path = "/data1/luofeng/RL/verl/data/sft_data/test_step.parquet"
    # process_2_parquet_sft_step(test_set, out_path)
#====================================rl带有step的数据处理================================================
    # # 读取jsonl文件，划分训练集和测试集为9:1
    # data_path = "/data1/luofeng/RL/verl/conduct_data/my_data_step.jsonl"
    # data = list(jsonlines.open(data_path, mode='r'))
    # print(f"Total samples: {len(data)}")
    # print(f"Example sample: {data[0]}")

    # data_rl = data[:3000]
    # train_num = math.floor(len(data_rl)*0.9)
    # train_set = data_rl[:train_num]
    # test_set = data_rl[train_num:]
    # print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    # output_dir = os.path.join("/data1/luofeng/RL/verl/data/rl_data", "train_step_small.parquet")
    # process_2_parquet_rl_step_small(train_set, output_dir)
    # output_dir = os.path.join("/data1/luofeng/RL/verl/data/rl_data", "test_step_small.parquet")
    # process_2_parquet_rl_step_small(test_set, output_dir)

