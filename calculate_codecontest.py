import os
import json
import sys
from test_one_solution import *
from collections import defaultdict
import gzip

root = "/home/jzchen/ML/LLMDebating/results"
exp_name = sys.argv[1]
start_idx = sys.argv[2]
end_idx = sys.argv[3]
item = sys.argv[4]
model_A_name = sys.argv[5]
model_B_name = sys.argv[6]

result_A = {}
result_B = {}


def convert(input_num):
    return str(input_num).zfill(4)


print(start_idx, end_idx)
for task_id in range(int(start_idx), int(end_idx) + 1):
    A_path = os.path.join(root, f"{exp_name}", f"{model_A_name}_{task_id}_{item}.json")
    B_path = os.path.join(root, f"{exp_name}", f"{model_B_name}_{task_id}_{item}.json")
    if not os.path.exists(A_path):
        A_path = os.path.join(
            root, f"{exp_name}", f"{model_A_name}_{task_id}_final.json"
        )
    if not os.path.exists(B_path):
        B_path = os.path.join(
            root, f"{exp_name}", f"{model_B_name}_{task_id}_final.json"
        )
    # result_A.update(json.load(open(A_path, "r")))
    # result_B.update(json.load(open(B_path, "r")))
    result_A[json.load(open(A_path, "r"))["task_id"]] = json.load(open(A_path, "r"))[
        "generation"
    ]
    result_B[json.load(open(B_path, "r"))["task_id"]] = json.load(open(B_path, "r"))[
        "generation"
    ]

result_A = {str(k): v for k, v in result_A.items()}
result_B = {str(k): v for k, v in result_B.items()}
print(len(result_A))
problems = defaultdict(dict)
data_path = f"/home/jzchen/ML/Multi-Teacher-Tree-Alignment/data/CodeContest/test/python_{exp_name.split('-')[-1]}.json.gz"
with gzip.open(data_path, "rb") as f:
    for line in f:
        line = eval(line)  # 执行一个表达式，并返回表达式的值
        problems[line["task_id"]] = (
            line  # fields:['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test']
        )
print("Model A results:")
eval_and_save_problems_from_code(result_A, problems)
print("Model B results:")
eval_and_save_problems_from_code(result_B, problems)
