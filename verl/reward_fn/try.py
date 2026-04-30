from openai import OpenAI
import pandas as pd
import re
from codebleu import calc_codebleu
import math
from typing import Any, Dict, Optional
import json
import jsonlines
from functools import lru_cache
import os

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

DATA_PATH = "/data1/luofeng/RL/verl/conduct_data/my_data_step.jsonl"
@lru_cache(maxsize=1)
def load_test_db():
    """
    每个 Python 进程只加载一次 JSONL，返回:
        { index(int): item(dict) }
    """
    db = {}
    with jsonlines.open(DATA_PATH, mode="r") as reader:
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



if __name__ == "__main__":
    #读取/data1/luofeng/RL/verl/data/sft_data/test_step_score.parquet，并把其中的prompt和answer打印出来
    df = pd.read_parquet("/data1/luofeng/RL/verl/data/sft_data/test_step_score.parquet")
    #打印指定行的prompt和answer
    row = df.iloc[-10]
    print("Prompt:")
    print(row["prompt"])
    print("Answer:")
    print(row["answer"])
    result = infer_one(row["prompt"], "Qwen2.5-Coder-7B-Instruct-sft-score")
    print("Model Output:")
    print(result)

    #=================测试那个只加载一次的函数是否能完成按照index索引数据的功能=================
    # index = 5
    # item = load_test_db().get(index)
    # c_code = item.get("c_code", "")
    # print("C Code:")
    # print(c_code)