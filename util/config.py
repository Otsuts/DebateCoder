import platform
import logging
import os
import openai


API_KEY = "YOUR GPT KEY"
base_url = "http://open.xiaoai.one/v1"


CLAUDE_KEY = "YOUR CLAUDE KEY"
CLAUDE_URL = "http://open.xiaoai.one/v1"

DEEPSEEK_KEY = "YOUR DEEPSEEK KEY"
DEEPSEEK_URL = "https://api.deepseek.com"


YI_KEY = "YOUR YI KEY"
YI_URL = "https://api.lingyiwanwu.com/v1"
QWEN_KEY = "YOUR QWEN KEY"
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
openai.api_key = API_KEY

logging.basicConfig(
    format="%(levelname)s %(asctime)s %(process)d %(message)s", level=logging.INFO
)

system = platform.system()
if system == "Linux":
    os.environ["http_proxy"] = "http://127.0.0.1:8888"
    os.environ["https_proxy"] = "http://127.0.0.1:8888"
    os.environ["all_proxy"] = "socks5://127.0.0.1:8889"
