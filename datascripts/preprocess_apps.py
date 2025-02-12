import json
import os
import random
import sys

level = sys.argv[1]
data_path = sys.argv[2]
if level == "intro":
    start_idx = 4001
    end_idx = 4100
elif level == "inter":
    start_idx = 1
    end_idx = 100
elif level == "comp":
    start_idx = 3001
    end_idx = 3100
else:
    raise NotImplementedError

test_size = 100
subpaths = os.listdir(data_path)
data = []
for task_id in sorted(subpaths):
    if int(task_id) >= start_idx and int(task_id) <= end_idx:
        question_path = os.path.join(data_path, task_id, "question.txt")
        with open(question_path, "r") as f:
            question = f.read()
        testcases_path = os.path.join(data_path, task_id, "input_output.json")
        testcases = json.load(open(testcases_path, "r"))
        testcases = [
            {"input": x[0], "output": x[1]}
            for x in zip(testcases["inputs"], testcases["outputs"])
        ]
        solutions = example_testcases = ""
        data.append(
            {
                "task_id": task_id,
                "prompt": question,
                "canonical_solution": solutions,
                "example_testcases": example_testcases,
                "testcases": testcases,
            }
        )

output_path = f"../data/apps_{level}.jsonl"
random.seed(42)
print(len(data))

with open(output_path, "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
