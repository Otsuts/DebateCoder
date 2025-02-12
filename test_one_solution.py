"""
Run solutions from one problem.
"""

import argparse
import json
import numpy as np
import os
import pprint
import multiprocessing
import util.testing_util as test_util

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from types import SimpleNamespace
from typing import Dict


EXAMPLE_RESULTS = {
    "0": [[-2]],
    "1": [[False, False, False]],
    "2": [[True, True]],
    "3": [
        [
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
        ]
    ],
    "4": [
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
    ],
}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 30


def print_results(results: Dict, args: argparse.Namespace = None):
    """
    Given the results evaluated against the testcases we output some statistics.

    >>> print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of test cases run = 5
    Test Case Average (average accuracy over problems) = 0.3
    Strict Accuracy (all test cases passed / total problems) = 0.2
    """
    res = []
    per_prob_res = []
    all_correct = []
    for index in results:
        problem_results = np.asarray(results[index])
        res.extend(problem_results)
        per_prob_res.append(np.mean(problem_results > 0))
        all_correct.append(np.all(problem_results > 0))

    # We count both compile errors and runtime errors for multiple tests as one error.
    compile_errors = len([e for e in res if -2 in e])
    runtime_errors = len([e for e in res if -1 in e])
    total_testcases = len(res)
    if args and args.debug:
        print(
            f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }"
        )
        print(
            f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}"
        )
        print(f"number of test cases run = {total_testcases}")

    print(
        f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}"
    )
    print(
        f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}"
    )


def check_correctness(
    prob_path, generation, timeout, debug, check_part="last", problem=None
):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    code = generation
    code = code.replace(":solution.py\n", "")
    if "__name__" in code and "__main__" in code:
        tmp_codes = code.split("\n")
        begin_main = False
        new_code = ""
        for line in tmp_codes:
            if begin_main:
                if line.startswith("\t"):
                    line = line[1:]
                elif line.startswith("    "):
                    line = line[4:]

            if not ("__name__" in line and "__main__" in line):
                new_code = new_code + line + "\n"
            else:
                begin_main = True

        generation = new_code

    def _temp_run(prob_path, generation, debug, result):
        result.append(
            test_util.run_test(
                prob_path=prob_path,
                test=generation,
                debug=debug,
                check_part=check_part,
                problem=problem,
            )
        )

    # result = []
    # _temp_run(prob_path, generation, debug, result)
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(
        target=_temp_run, args=(prob_path, generation, debug, result)
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"global timeout")
    return result[0]


def eval_and_save_problems_from_code(code_dict, all_problems=None):
    root = "../Code/data/APPS/test/"
    gpt_codes = code_dict

    problems = list(gpt_codes.keys())
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}

    # main eval loop
    for problem_idx in tqdm(problems):
        problem_idx = str(problem_idx)
        output_str = gpt_codes[problem_idx]
        prob_path = os.path.join(root, problem_idx)
        res = []
        curr_res = [-2]
        try:
            curr_res = check_correctness(
                prob_path=prob_path,
                generation=output_str,
                timeout=TIMEOUT,
                debug=False,
                problem=all_problems[int(problem_idx)],
            )
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            # if not np.all(curr_res):
            #     print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)

        results[problem_idx] = res
    print_results(results)

    return results


def convert(input_num):
    return str(input_num).zfill(4)


def eval_and_save_problems(args):
    codes_loc = args.test_loc
    gpt_codes = {}
    with open(codes_loc, "r") as f:
        gpt_codes = json.load(f)
    gpt_codes = {convert(k): v for k, v in gpt_codes.items() if int(k) != 0}

    problems = list(gpt_codes.keys())
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}

    results_loc = os.path.join(args.save, "results.json")
    print(codes_loc, results_loc)

    if args.index:
        problems = [str(args.index)]

    # main eval loop
    for problem_idx in tqdm(problems):
        output_str = gpt_codes[problem_idx]
        prob_path = os.path.join(args.root, problem_idx)

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        res = []

        if args.debug:
            print(f"\nTesting solution {problem_idx}")
        curr_res = [-2]
        try:
            curr_res = check_correctness(
                prob_path=prob_path,
                generation=output_str,
                timeout=TIMEOUT,
                debug=args.debug,
            )
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)

        if args.debug:
            print(
                f"\nHow to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case"
            )
            # print(f"results = {res}")

        results[problem_idx] = res
        if args.save_results:
            with open(results_loc, "w") as f:
                try:
                    f.write(json.dumps(results))
                except Exception as e:
                    import pdb

                    pdb.set_trace()
                    print("didn't save problem due to {e}")

    return results


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    results = eval_and_save_problems(args)

    print_results(results, args)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    parser = argparse.ArgumentParser(
        description="Testing a Language Model on Python Code"
    )
    parser.add_argument(
        "-t",
        "--test_loc",
        default="/home/jzchen/ML/Code/output/gemma-7b-it_appsnew.json",
        type=str,
        help="path to the json containing problem paths to be evaluated.",
    )
    parser.add_argument(
        "-r",
        "--root",
        default="../Code/data/APPS/test/",
        type=str,
        help="where the apps dataset is stored.",
    )
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument(
        "--save",
        type=str,
        default="./output/",
        help="Where the evaluated data is loaded from and results saved to.",
    )
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save_results", "-s", action="store_true")

    args = parser.parse_args()

    main(args)
