import argparse
import sys
from peft import LoraConfig
import warnings
from codegeex.data.data_utils import write_jsonl
import json
import transformers
import random

random.seed(42)

sys.path.append("..")
from models.agent import LLMAgent
from models.coordinator import Cordinator
from util.config import *
from dataloaders.datadealer import DataDealer


def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--model_A_name", "-A", type=str)
    parser.add_argument("--model_B_name", "-B", type=str)
    parser.add_argument("--model_A_device", "-a", type=str)
    parser.add_argument("--model_B_device", "-b", type=str)
    # Dataset parameters
    parser.add_argument("--language", "-l", type=str, choices=["python"])
    parser.add_argument("--dataset", "-d", type=str, choices=["apps", "CodeContest"])
    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--level", type=str, default="simple")
    # Running parameters
    parser.add_argument("--root_result_path", type=str, default="../results/")
    parser.add_argument("--cur_exp_name", type=str)
    parser.add_argument(
        "--data_root_path", type=str, default="/home/jzchen/ML/Code/data/APPS/test"
    )
    parser.add_argument(
        "--times", "-t", type=int, default=1, help="running time (for multiple run)"
    )
    parser.add_argument("--start_idx", type=int, default=4001)
    parser.add_argument("--end_idx", type=int, default=4100)
    parser.add_argument(
        "--debating_model",
        type=str,
        default="both",
        choices=[
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "deepseek-chat",
            "both",
            "yi-34b-chat-0205",
            "qwen-plus",
        ],
    )
    parser.add_argument("--analyse", action="store_true")

    return parser.parse_args()


def get_dataset(args):

    datadealer = DataDealer(args.dataset, level=args.level)
    return datadealer


def get_agent(args):
    peft_config = LoraConfig(
        # task_type=TaskType.CAUSAL_LM,
        # inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    agent1 = LLMAgent(
        model=args.model_A_name,
        device=args.model_A_device,
        language=args.language,
        data_root_path=args.data_root_path,
    )
    agent2 = LLMAgent(
        model=args.model_B_name,
        device=args.model_B_device,
        language=args.language,
        data_root_path=args.data_root_path,
    )
    return agent1, agent2


def update_one_epoch(
    language: str,
    task_id: str,
    question: str,
    agent1: LLMAgent,
    agent2: LLMAgent,
    solution1=None,
    solution2=None,
    epoch=0,
    problem=None,
):
    assert solution1 and solution2
    # Step1: test case generation
    test_case1 = agent1.testcase_generate(
        language, task_id, question, solution1, solution2, problem
    )
    test_case2 = agent2.testcase_generate(
        language, task_id, question, solution2, solution1, problem
    )
    # Step2: solution update
    update_solA = agent1.update_solution(
        language, question, test_case2, solution2, solution1
    )
    update_solB = agent2.update_solution(
        language, question, test_case1, solution1, solution2
    )
    return update_solA, update_solB


def update_one_epoch_analysis(
    language: str,
    task_id: str,
    question: str,
    agent1: LLMAgent,
    agent2: LLMAgent,
    solution1,
    solution2,
    problem=None,
):
    assert solution1 and solution2
    # solution update
    update_solA = agent1.update_solution_analysis(
        language, question, None, solution2, solution1
    )
    update_solB = agent2.update_solution_analysis(
        language, question, None, solution1, solution2
    )
    return update_solA, update_solB


def main(args):
    os.makedirs(os.path.join(args.root_result_path, args.cur_exp_name), exist_ok=True)
    agent1, agent2 = get_agent(args)
    datadealer = get_dataset(args)
    for task_id, problem in datadealer.iter_test_data():
        if args.debating_model != "both":
            agent1.model_name = args.model_A_name
            agent2.model_name = args.model_B_name

        if int(task_id) not in range(
            args.start_idx, args.end_idx + 1
        ):  # Not in wanted index range
            continue
        question = problem["prompt"]
        # Zeroshot results
        solA = agent1.solution_generate(args.language, question)
        solB = agent2.solution_generate(args.language, question)
        model_A_cur, model_A_all = agent1.get_cur_results(
            str(task_id), solA, check_part="first", problem=problem
        )
        model_B_cur, model_B_all = agent2.get_cur_results(
            str(task_id), solB, check_part="first", problem=problem
        )
        # New code passed all testcases
        if model_A_cur == model_A_all and model_B_cur == model_B_all:
            print(f"All correct at zero shot.")
            datadealer.save_results(
                [datadealer.formatted_return(solA, problem, task_id)],
                os.path.join(
                    args.root_result_path,
                    args.cur_exp_name,
                    f"{args.model_A_name}_{task_id}_final.json",
                ),
            )
            datadealer.save_results(
                [datadealer.formatted_return(solB, problem, task_id)],
                os.path.join(
                    args.root_result_path,
                    args.cur_exp_name,
                    f"{args.model_B_name}_{task_id}_final.json",
                ),
            )
            continue

        datadealer.save_results(
            [datadealer.formatted_return(solA, problem, task_id)],
            os.path.join(
                args.root_result_path,
                args.cur_exp_name,
                f"{args.model_A_name}_{task_id}_zeroshot.json",
            ),
        )
        datadealer.save_results(
            [datadealer.formatted_return(solB, problem, task_id)],
            os.path.join(
                args.root_result_path,
                args.cur_exp_name,
                f"{args.model_B_name}_{task_id}_zeroshot.json",
            ),
        )

        # Self envolvement
        # Update according to the demo testcases
        try:
            solA5 = agent1.update_via_demo(
                args.language, str(task_id), question, solA, problem_all=problem
            )
            solB5 = agent1.update_via_demo(
                args.language, str(task_id), question, solB, problem_all=problem
            )
        except:
            solA5 = solA
            solB5 = solB

        if (
            agent1.get_cur_results(
                str(task_id), solA5, check_part="first", problem=problem
            )[0]
            >= agent1.get_cur_results(
                str(task_id), solA, check_part="first", problem=problem
            )[0]
        ):
            solA = solA5
        if (
            agent2.get_cur_results(
                str(task_id), solB5, check_part="first", problem=problem
            )[0]
            >= agent1.get_cur_results(
                str(task_id), solB, check_part="first", problem=problem
            )[0]
        ):
            solB = solB5
        # Save results
        datadealer.save_results(
            [datadealer.formatted_return(solA, problem, task_id)],
            os.path.join(
                args.root_result_path,
                args.cur_exp_name,
                f"{args.model_A_name}_{task_id}_selfevolve.json",
            ),
        )
        datadealer.save_results(
            [datadealer.formatted_return(solB, problem, task_id)],
            os.path.join(
                args.root_result_path,
                args.cur_exp_name,
                f"{args.model_B_name}_{task_id}_selfevolve.json",
            ),
        )
        if args.debating_model != "both":
            agent1.model_name = args.debating_model
            agent2.model_name = args.debating_model

        # Model debating
        stop = 0
        model_A_correct = 0
        model_B_correct = 0
        for epoch in range(1, args.epochs + 1):
            try:
                if args.analyse:
                    solA2, solB2 = update_one_epoch_analysis(
                        args.language,
                        str(task_id),
                        question,
                        agent1,
                        agent2,
                        solA,
                        solB,
                        epoch=epoch - 1,
                        problem=problem,
                    )
                else:
                    solA2, solB2 = update_one_epoch(
                        args.language,
                        str(task_id),
                        question,
                        agent1,
                        agent2,
                        solA,
                        solB,
                        epoch=epoch - 1,
                        problem=problem,
                    )
                model_A_cur, model_A_all = agent1.get_cur_results(
                    str(task_id), solA2, check_part="first", problem=problem
                )
                model_B_cur, model_B_all = agent2.get_cur_results(
                    str(task_id), solB2, check_part="first", problem=problem
                )
                # New code passed all testcases
                if model_A_cur == model_A_all or model_B_cur == model_B_all:
                    print(f"All correct at epoch {epoch}.")
                    solA = solA2
                    solB = solB2
                    break
                # No furthur progress, early stop ++
                if model_A_cur <= model_A_correct and model_B_cur <= model_B_correct:
                    stop += 1
                    if stop >= args.early_stop:
                        print(f"Early stop at {epoch} epoch.")
                        break
                model_A_correct = max(model_A_correct, model_A_cur)
                model_B_correct = max(model_B_correct, model_B_cur)

                if (
                    agent1.get_cur_results(
                        str(task_id), solA2, check_part="first", problem=problem
                    )[0]
                    >= agent1.get_cur_results(
                        str(task_id), solA, check_part="first", problem=problem
                    )[0]
                ):
                    solA = solA2
                if (
                    agent2.get_cur_results(
                        str(task_id), solB2, check_part="first", problem=problem
                    )[0]
                    >= agent1.get_cur_results(
                        str(task_id), solB, check_part="first", problem=problem
                    )[0]
                ):
                    solB = solB2
            except Exception as e:
                print("*" * 10, "Fuck!! Something wrong happened", "*" * 10)
                print(e)

            datadealer.save_results(
                [datadealer.formatted_return(solA, problem, task_id)],
                os.path.join(
                    args.root_result_path,
                    args.cur_exp_name,
                    f"{args.model_A_name}_{task_id}_{epoch}.json",
                ),
            )
            datadealer.save_results(
                [datadealer.formatted_return(solB, problem, task_id)],
                os.path.join(
                    args.root_result_path,
                    args.cur_exp_name,
                    f"{args.model_B_name}_{task_id}_{epoch}.json",
                ),
            )

            print(
                "*" * 10,
                f"Problem {task_id}, epoch {epoch}: ",
                agent1.get_cur_results(
                    task_id, solA, check_part="all", problem=problem
                ),
                "*" * 10,
            )
        agent1.reset()
        agent2.reset()
        datadealer.save_results(
            [datadealer.formatted_return(solA, problem, task_id)],
            os.path.join(
                args.root_result_path,
                args.cur_exp_name,
                f"{args.model_A_name}_{task_id}_final.json",
            ),
        )
        datadealer.save_results(
            [datadealer.formatted_return(solB, problem, task_id)],
            os.path.join(
                args.root_result_path,
                args.cur_exp_name,
                f"{args.model_B_name}_{task_id}_final.json",
            ),
        )


if __name__ == "__main__":
    transformers.set_seed(42)
    args = get_args()
    main(args)
    exit()
