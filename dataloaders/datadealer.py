import json
import gzip
import os
from collections import defaultdict
from datasets import load_dataset
from codegeex.data.data_utils import write_jsonl
import sys
import re
import pickle as pkl
import warnings

sys.path.append("../")
import numpy as np
from tqdm import tqdm


class DataDealer:
    def __init__(self, dataset, cutofflen=3600, level="simple"):

        self.cutofflen = cutofflen
        self.dataset = dataset
        if dataset == "apps":
            self.data_path = f"../apps_{level}.jsonl.gz"
        elif dataset == "CodeContest":
            self.data_path = f"../CodeContest/python_{level}.json.gz"
        else:
            print(f"Dataset {dataset} is not supported.")
            raise NotImplementedError

        self.problems = None
        self.language = "python"
        self.load_data()

    def get_datapath(self):
        return self.data_path

    def load_data(self):
        if self.data_path.endswith(".json"):
            self.problems = json.load(open(self.data_path, "r"))
        elif self.data_path.endswith(".gz"):
            self.problems = defaultdict(dict)
            with gzip.open(self.data_path, "rb") as f:
                for line in f:
                    line = eval(line)  #
                    self.problems[line["task_id"]] = (
                        line  # fields:['task_id', 'prompt', 'canonical_solution', 'test', 'text', 'declaration', 'example_test']
                    )

    def save_as_json(self, object, path):
        with open(
            path,
            "w",
        ) as f:
            json.dump(object, f)

    def iter_test_data(self):
        print(f"Total dataset length {len(self.problems)}")
        for task_id in tqdm(self.problems):
            problem = self.problems[task_id]
            yield task_id, problem

    def formatted_return(self, completion, problem, task_id):
        ans = {}
        if "apps" in self.dataset:  # Apps Dataset
            ans[int(task_id)] = completion
        else:  # CodeForce Dataset
            ans = dict(
                task_id=task_id,
                generation=completion,
                prompt=problem["prompt"],
                example_testcases=problem["example_testcases"],
                testcases=problem["testcases"],
            )
        return ans

    def save_results(self, object, path):
        if "apps" in self.dataset:
            if path.endswith("jsonl"):
                warnings.warn("apps dataset requires a .json file, renaming path")
                path = path.replace("jsonl", "json")
            samples = {}
            for ans in object:
                samples[list(ans.keys())[0]] = list(ans.values())[0]
            self.save_as_json(samples, path)
        else:
            write_jsonl(path, object)
        print("Results saved at", path)
