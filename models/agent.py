import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import re
import json
from openai import OpenAI
from util.config import *
from util.feedback import human_feedback
from test_one_solution import check_correctness
import numpy as np
from vllm import LLM, SamplingParams
import warnings
import gc
import random
import time

warnings.filterwarnings("ignore")


class LLMAgent:

    def __init__(
        self,
        model,
        max_length=1000,
        device="0",
        language="python",
        data_root_path="/home/jzchen/ML/Code/data/APPS/test",
    ):
        assert model in [
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "deepseek-chat",
            "yi-34b-chat-0205",
            "qwen-plus",
        ]
        self.model_name = model
        self.model = None
        self.tokenizer = None
        self.max_length = max_length
        self.device = f"cuda:{device}"
        self.language = language
        self.data_root_path = data_root_path
        self.testcases = []

        # Load Model
        self.load_model()

    def reset(self):
        self.testcases.clear()
        gc.collect()

    def solution_generate(self, language: str, task: str) -> str:
        input_text = f"Please use {language} to write a correct solution to a programming problem. You should give executable completed code and nothing else. The problem:\n{task}"
        # input_text = f"Complete the {language} program to solve the problem. Remember to contain the complete program including all the imports and function header in your response.\n Generate the code ONLY. No other explanation or words attached!\n The problem:\n{task}"
        return self.generate(input_text)

    def extract_demo_testcases(self, idx: str, problem_all: dict):
        if problem_all.get("example_testcases", 0):
            return [
                {"input": i, "output": o}
                for i, o in zip(
                    problem_all["example_testcases"]["input"],
                    problem_all["example_testcases"]["output"],
                )
            ]
        question_path = os.path.join(self.data_root_path, idx, "question.txt")
        with open(question_path, "r", encoding="utf-8") as f:
            problem = f.read()
        problem += "\n\n"
        # Extract input and output using re module
        examples = re.findall(r"Input\n(.*?)\n\nOutput\n(.*?)\n\n", problem, re.DOTALL)
        if not examples:
            examples = re.findall(
                r"Sample Input \d+:\n(.*?)\nSample Output \d+:\n(.*?)\n\n",
                problem,
                re.DOTALL,
            )
        if not examples:
            examples = re.findall(
                r"-----Sample Input-----\n(.*?)\n\n-----Sample Output-----\n(.*?)\n\n",
                problem,
                re.DOTALL,
            )
        if not examples:
            print(problem)
        # Construct dict
        testcases = [
            {"input": inp.strip() + "/n", "output": out.strip() + "/n"}
            for inp, out in examples
        ]
        return testcases

    def calculate_demo_score(self, idx, current_solution):
        testcases = self.extract_demo_testcases(idx)
        failed_num = 0
        for testcase in testcases:
            input, output = testcase["input"], testcase["output"]
            output, execution_results = human_feedback(current_solution, input, output)
            failed_num += output != execution_results
        return failed_num

    def testcase_based_update(self, language, problem, current_solution):
        print(f"Using {len(self.testcases)} testcases to update.")
        failed_cases = []
        for testcase in self.testcases:
            input, output = testcase["input"], testcase["output"]
            output, execution_results = human_feedback(current_solution, input, output)
            if output != execution_results:
                failed_cases.append(
                    {
                        "input": input,
                        "expected output": output,
                        "your output": execution_results,
                    }
                )
        if not failed_cases:
            return current_solution
        failed_cases = json.dumps(failed_cases, ensure_ascii=False, indent=4)
        prompt = f"You are a skilled programmer. Given a {language} programming problem and your current solution. We execute the solution on some demo testcases and get the results along with the correct answer. Please analyse the results and give useful advice to improve the solution to pass the testcases. The problem: {problem}. The current solution: {current_solution}. Execution results: {failed_cases}. Give me your advice and improved code."
        advice = self.generate(prompt, testcase=True)
        prompt = f"Given a {language} programming problem and your current solution. An expert give some advice on how to improve it. Please make the solution correct and faster. The problem:{problem}. The current solution: {current_solution}. The expert suggestions: {advice}. Give me ONLY the improved code and nothing else."
        output = self.generate(prompt)
        return output

    def update_via_demo(self, language, idx, problem, current_solution, problem_all):
        testcases = self.extract_demo_testcases(idx, problem_all)
        print(f"demo has {len(testcases)} testcases.")
        failed_cases = []
        for testcase in testcases:
            input, output = testcase["input"], testcase["output"]
            output, execution_results = human_feedback(current_solution, input, output)
            if output != execution_results:
                failed_cases.append(
                    {
                        "input": input,
                        "expected output": output,
                        "your output": execution_results,
                    }
                )
        if not failed_cases:
            return current_solution
        failed_cases = json.dumps(failed_cases, ensure_ascii=False, indent=4)
        prompt = f"You are a skilled programmer. Given a {language} programming problem and your current solution. We execute the solution on some demo testcases and get the results along with the correct answer. Please analyse the results and give useful advice to improve the solution to pass the testcases. The problem: {problem}. The current solution: {current_solution}. Execution results: {failed_cases}. Give me your advice and improved code."
        advice = self.generate(prompt, testcase=True)
        prompt = f"Given a {language} programming problem and your current solution. An expert give some advice on how to improve it. Please make the solution correct and faster. The problem:{problem}. The current solution: {current_solution}. The expert suggestions: {advice}. Give me ONLY the improved code and nothing else."
        output = self.generate(prompt)
        return output

    def testcase_generate(
        self,
        language: str,
        task_id: str,
        task: str,
        correct_sol: str,
        wrong_sol: str,
        problem: dict,
    ):
        """
        Test case generation for model debating:
        1. Testcase generation
        2. Revise testcases
        """
        testcases = self.extract_demo_testcases(task_id, problem)
        demo_testcase_input = []
        for testcase in testcases:
            demo_testcase_input.append({"Input": testcase["input"]})
        if not testcases:
            demo_testcase_input.append(
                {
                    "Input": "3\n3 10\n6 3\n8 2\n1 4\n4 10\n4 1\n3 2\n2 6\n1 100\n2 15\n10 11\n14 100\n"
                }
            )
        input_text = f"You are an excellent programmer. I'll give you a {language} programming problem and an imperfect solution. Your task is to analyse the imperfect code, and generate one test case that will fail the solution. Your test case must be in a json format, with the input being a string. An example of test case is: {demo_testcase_input[0]}. The problem: {task}. Solution to be improve: {wrong_sol}. Just give me your test case and no other explanations."
        testcase_string = self.generate(input_text, testcase=True)
        json_string = self.testcase_filter(testcase_string).strip()
        data = json.loads(json_string)
        input_string = data.get("Input", "")  # testcase input
        _, output_string = human_feedback(correct_sol, input_string)
        # Improve testcase
        if (
            "Error" in output_string or "No Output" in output_string
        ):  # test case did'n pass the model A's script
            revice_prompt = f"Given a {language} code, your testcase for it is {json_string}, which yeild the error {output_string}. Please debug the testcase. Your test case must be in a json format, with the input being a string. Just give me your improved test case and no other explanations."
            # Generate a new testcase
            testcase_string = self.generate(revice_prompt, testcase=True)
            json_string = self.testcase_filter(testcase_string).strip()
            data = json.loads(json_string)
            input_string = data.get("Input", "")
            _, output_string = human_feedback(correct_sol, input_string)
            if "Error" in output_string:
                print(f"{self.model_name} did not make testcase right!")
                cur_testcase = random.choice(testcases)
                return json.dumps(
                    {
                        "input": cur_testcase["input"],
                        "output": cur_testcase["output"],
                    }
                )
        generated_testcase = {"input": input_string, "output": output_string}
        return json.dumps(generated_testcase)

    def analyse_and_advice(self, language: str, task: str, current_sol: str) -> str:
        input_test = f"You are an excellent programmer. I'll give you a {language} programming problem and an imperfect solution. Your task is to give me useful advice on how to make it correct. The problem: {task}. Solution to be improve: {current_sol}. Just give me your advice and nothing else."
        advice = self.generate(input_test, testcase=True)
        return advice

    def update_solution(
        self,
        language: str,
        task: str,
        test_cases: str,
        correct_sol: str,
        current_sol: str,
        epoch: int,
    ):
        self.testcases.append(json.loads(test_cases))
        print(f"Using {len(self.testcases)} testcases to update.")
        failed_cases = []
        for testcase in self.testcases:
            input, output = testcase["input"], testcase["output"]
            output, execution_results = human_feedback(current_sol, input, output)
            if output != execution_results:
                failed_cases.append(
                    {
                        "input": input,
                        "another output": output,
                        "your output": execution_results,
                    }
                )
                print(failed_cases)
        if not failed_cases:
            print("Passed all testcases, use analyse to improve")
            # Step1: Analyse the code
            prompt = f"Given a {language} programming problem and your current solution, we also have another person's code for you to compare yours. Please give useful advice along with your correct code to improve the solution to make it correct and better. The problem:{task}. The current solution: {current_sol}. Another person's solution: {correct_sol}. "
            advice = self.generate(prompt, testcase=True)

            # Step2: Improve the code
            prompt = f"Given a {language} programming problem and your current solution. We also have the solution by another person. An expert give some advice on how to improve it. Please improve your current solution base on the advice. The problem:{task}. The current solution: {current_sol}. Another solution: {correct_sol}. The expert suggestions: {advice}. Give me ONLY the improved code and nothing else."
            output = self.generate(prompt)
            return output
        failed_cases = json.dumps(failed_cases, ensure_ascii=False, indent=4)

        # Step2: Analyse the code
        prompt = f"Given a {language} programming problem, a current solution and several testcases, we execute the solution on the testcase and get the results. Also we have another person's code and his execution results for you to compare yours. Your task: 1. Briefly summarize the deficiencies in the current solution, 2. Give useful advice along with your correct code to improve the solution to make it correct and better. The problem:{task}. The current solution: {current_sol}. Another person's solution: {correct_sol}. Execution results:{failed_cases}"
        advice = self.generate(prompt, testcase=True)

        # Step3: Improve the code
        prompt = f"Given a {language} programming problem and your current solution. We also have the solution by another person. An expert give some advice on how to improve it. Please improve your current solution base on the advice. The problem:{task}. The current solution: {current_sol}. Another solution: {correct_sol}. The expert suggestions: {advice}. Give me ONLY the improved code and nothing else."
        output = self.generate(prompt)
        return output

    def update_solution_analysis(
        self,
        language: str,
        task: str,
        test_cases: str,
        correct_sol: str,
        current_sol: str,
    ) -> str:

        # Step1: Analyse the code
        prompt = f"Given a {language} programming problem and your current solution, we also have another person's code for you to compare yours. Please give useful advice along with your correct code to improve the solution to make it correct and better. The problem:{task}. The current solution: {current_sol}. Another person's solution: {correct_sol}. "
        advice = self.generate(prompt, testcase=True)

        # Step3: Improve the code
        prompt = f"Given a {language} programming problem and your current solution. We also have the solution by another person. An expert give some advice on how to improve it. Please improve your current solution base on the advice. The problem:{task}. The current solution: {current_sol}. Another solution: {correct_sol}. The expert suggestions: {advice}. Give me ONLY the improved code and nothing else."
        output = self.generate(prompt)
        return output

    def generate(self, text: str, testcase=False):
        """
        text: Question prompt
        return: Model generated answer
        """
        user_prompt = self.construct_prompt(text)
        if self.model_name == "deepseek-chat":
            for ti in range(20):
                try:
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": user_prompt}],
                        stream=False,
                    )
                    return self.filt_output(
                        response.choices[0].message.content, testcase=testcase
                    )
                except:
                    print(f"Try {ti+1} times")
                    time.sleep(7 * ti)
        elif self.model_name == "yi-34b-chat-0205":
            for ti in range(20):
                try:
                    response = self.client.chat.completions.create(
                        model="yi-34b-chat-0205",
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    return self.filt_output(
                        response.choices[0].message.content, testcase=testcase
                    )
                except:
                    print(f"Try {ti+1} times")
                    time.sleep(7 * ti)
        elif self.model_name == "qwen-plus":
            for ti in range(20):
                try:
                    response = self.client.chat.completions.create(
                        model="qwen-plus",
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    return self.filt_output(
                        response.choices[0].message.content, testcase=testcase
                    )
                except:
                    print(f"Try {ti+1} times")
                    time.sleep(7 * ti)
        elif "claude" in self.model_name:
            self.client.api_key = CLAUDE_KEY
            self.client.base_url = CLAUDE_URL
            for ti in range(20):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": user_prompt}],
                        max_tokens=self.max_length,
                        temperature=0.0,
                        top_p=0.0,
                        seed=42,
                    )
                    return self.filt_output(
                        response.choices[0].message.content, testcase=testcase
                    )
                except:
                    print(f"Try {ti+1} times")
                    time.sleep(7 * ti)
        else:  # GPT
            for ti in range(20):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": user_prompt}],
                        max_tokens=self.max_length,
                        temperature=0.0,
                        top_p=0.0,
                        seed=42,
                    )
                    return self.filt_output(
                        response.choices[0].message.content, testcase=testcase
                    )
                except:
                    print(f"Try {ti+1} times")
                    time.sleep(7 * ti)

    def decode_json_io(self, json_string):
        data = json.loads(json_string)
        input_string = data.get("Input", "")
        output_string = data.get("Output", "")
        return input_string, output_string

    def get_cur_results(self, idx, generation, check_part="last", problem=None):
        prob_path = os.path.join(self.data_root_path, str(idx))
        cur_res = check_correctness(
            prob_path=prob_path,
            generation=generation,
            timeout=30,
            debug=False,
            check_part=check_part,
            problem=problem,
        )
        fixed = []
        for e in cur_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
        return sum(curr_res), len(curr_res)

    def testcase_filter(self, testcase: str) -> str:
        pattern = rf"```json(.*?)```"
        matches = re.findall(pattern, testcase, re.DOTALL)
        if matches:
            testcase = matches[0].strip()
        return testcase.replace("'", '"')  # Json string
        idx = testcase.find("'Input: input_data, Output: output_data'")
        testcase = testcase[idx + len("'Input: input_data, Output: output_data'") :]
        idx = testcase.find("Input")
        testcase = testcase[idx:]
        idx = testcase.find("assistant")
        if idx == -1:
            return testcase
        return testcase[:idx]

    def load_model(
        self,
    ):
        """
        Load model according to different model types (disk LLM, GPT)
        """

        print(f"initialize{self.model_name}, {self.device}")
        if "gpt" in self.model_name:  # GPT API
            self.client = OpenAI(api_key=API_KEY)
            self.client.base_url = base_url
        elif "claude" in self.model_name:  # GPT API
            self.client = OpenAI(api_key=CLAUDE_KEY)
            self.client.base_url = CLAUDE_URL
        elif self.model_name == "deepseek-chat":
            self.client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url=DEEPSEEK_URL,
            )
        elif "yi" in self.model_name:  # Yi
            self.client = OpenAI(
                api_key=YI_KEY,
                base_url=YI_URL,
            )
        else:  # Qwen
            self.client = OpenAI(api_key=QWEN_KEY, base_url=QWEN_URL)

        print(f"Model {self.model_name} initialized")

    def construct_prompt(self, text: str) -> str:
        return text  # GPT

    def filt_output(self, output: str, testcase: bool):
        if testcase:
            return output
        completion = output
        if (
            "gpt" in self.model_name
            or "claude" in self.model_name
            or "deepseek-chat" in self.model_name
            or "yi" in self.model_name
            or "qwen-plus" in self.model_name
        ):
            pattern = rf"```{self.language.replace('++', 'pp')}(.*?)```"
            matches = re.findall(pattern, output, re.DOTALL)
            if matches:
                completion = matches[0].strip()
        return completion
