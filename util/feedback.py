from util.repl import PythonREPL
import sys
from io import StringIO


def run(testcases: str, scripts: str):
    """
    Tool function given LLM
    """
    # 保存原始的标准输入
    original_stdin = sys.stdin

    # 创建一个字符串流，模拟用户输入
    input_data = testcases
    sys.stdin = StringIO(input_data)
    a = PythonREPL({})
    output = a(scripts)
    # 恢复原始的标准输入
    sys.stdin = original_stdin
    del a
    return output


def human_feedback(python_script, testcases, output=None):
    # Output indicates expected output
    model_output = run(testcases, python_script)
    if "Error" in model_output and not output:  # Illegal testcase
        print("illegal testcases")
    if output:
        output = output.strip()
    return output, model_output.strip()


if __name__ == "__main__":
    # Given by LLM
    python_script = "for _ in range(int(input())):\n    n, x = list(map(int, input().split()))\n    A = []\n    for _1 in range(n):\n       d, h = list(map(int, input().split()))\n       A.append([d, h])\n    A.sort(reverse=True)\n    if A[0][0] >= x:\n        print(1)\n    else:\n        x -= A[0][0]\n        mz = 0\n        for d, h in A:\n            mz = max(mz, d - h)\n        if mz:\n            print((x + mz - 1) // mz + 1)\n        else:\n            print(-1)\n"
    testcases = (
        "3\n3 10\n6 3\n8 2\n1 4\n4 10\n4 1\n3 2\n2 6\n1 100\n2 15\n10 11\n14 100\n"
    )
    output = "2\n3\n-1\n"

    # Call the feedback function
    print(human_feedback(python_script, testcases, output))
