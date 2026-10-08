"""DeepSeek ChatModel：初始化与 invoke / stream / batch 调用示例。

DeepSeek 提供 OpenAI 兼容接口，因此通过 model_provider="openai" + base_url 接入。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def build_chat_model():
    """初始化 ChatModel。

    等价写法（直接实例化提供商类）：
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model="deepseek-chat",
            base_url=DEEPSEEK_BASE_URL,
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请先在 .env 中填写真实的 API Key")

    return init_chat_model(
        "deepseek-chat",  # deepseek-chat = V3 通用模型；deepseek-reasoner = R1 推理模型
        model_provider="openai",
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        temperature=0.7,
        max_tokens=1024,
        timeout=60,
        max_retries=2,
    )


def main() -> None:
    model = build_chat_model()

    # 1. 单次同步调用
    response = model.invoke("你好，请用一句话介绍你自己")
    print("[invoke]", response.content)

    # 2. 流式输出（逐 token 返回，适合聊天界面）
    print("[stream] ", end="")
    for chunk in model.stream("写一首关于春天的五言绝句"):
        print(chunk.content, end="", flush=True)
    print()

    # 3. 批量调用（并行处理多个请求）
    responses = model.batch(["1+1 等于几？只回答数字", "中国的首都是哪里？只回答城市名"])
    for i, resp in enumerate(responses, 1):
        print(f"[batch {i}] {resp.content}")


if __name__ == "__main__":
    main()