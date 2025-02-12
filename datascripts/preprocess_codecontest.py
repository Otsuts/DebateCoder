import json
import fire
import gzip
import os
import pandas as pd
from pprint import pprint
import pickle as pkl
import random
import numpy as np


def prompt_filter(datapoint):
    # example_idx = datapoint.find("Examples")
    # datapoint = datapoint[:example_idx]
    return datapoint


def solution_filter_cpp(datapoint):
    languages = datapoint["language"]
    solutions = datapoint["solution"][languages == 2]
    if len(solutions):
        return min(solutions, key=lambda x: len(x))
    else:
        return None


def solution_filter_python(datapoint):
    languages = datapoint["language"]
    solutions = datapoint["solution"][languages == 3]
    if len(solutions):
        return min(solutions, key=lambda x: len(x))
    else:
        return None


def main(
    language="python",
    data_path="../data/",
    dataset="CodeContest",
    difficulty="hard",
    data_root_path="/home/jzchen/ML/Code/data/CodeContest",
    problem_num=100,
):
    problem_file = data_root_path
    fp = os.path.join(data_path, f"{dataset}/test/", f"{language}_{difficulty}.json")
    os.makedirs(os.path.join(data_path, f"{dataset}/test"), exist_ok=True)
    problems = []
    if difficulty == "easy":
        diff_range = range(0, 11)
    else:
        diff_range = range(11, 26)
    cur_idx = 0
    for filename in os.listdir(problem_file):
        if filename.startswith("test") or filename.startswith("valid"):
            df = pd.read_parquet(os.path.join(problem_file, filename))
            df["description"] = df["description"].apply(lambda x: prompt_filter(x))
            df["public_tests"] = df["public_tests"].apply(
                lambda x: {"input": x["input"].tolist(), "output": x["output"].tolist()}
            )
            df["private_tests"] = df["private_tests"].apply(
                lambda x: {"input": x["input"].tolist(), "output": x["output"].tolist()}
            )
            if language == "c++":
                df["solutions"] = df["solutions"].apply(
                    lambda x: solution_filter_cpp(x)
                )
            elif language == "python":
                df["solutions"] = df["solutions"].apply(
                    lambda x: solution_filter_python(x)
                )
            elif language == "java":
                pass
            else:
                raise NotImplementedError
            for idx, datapoint in df.iterrows():
                if not datapoint["solutions"]:
                    continue
                cur_datapoint = {
                    "task_id": cur_idx,
                    "prompt": datapoint["description"],
                    "canonical_solution": datapoint["solutions"],
                    "example_testcases": datapoint["public_tests"],
                    "testcases": datapoint["private_tests"],
                    "difficulty": datapoint["difficulty"],
                }

                if int(cur_datapoint["difficulty"]) in diff_range:
                    problems.append(cur_datapoint)
                    cur_idx += 1
                # for k, v in problems[0].items():
                #     print(k, type(v))
                # assert 0
    print(f"Dataset lenth {len(problems)}")
    # problems = random.sample(problems, min(problem_num, len(problems)))
    with open(fp, "w") as f:
        for item in problems:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    fire.Fire(main)


"""
# Usage
### Load Data
problems = defaultdict(dict)
with gzip.open(self.data_path, "rb") as f:
    for line in f:
        line = eval(line)  # 执行一个表达式，并返回表达式的值
        problems[line["task_id"]] = (
            line  # fields:['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test']
        )
### Iter Data
for task_id in tqdm(problems):
    [YOUR CODE HERE]
"""
