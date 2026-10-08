"""Memory 最小示例：让 Agent 记住同一个会话的历史。

核心是 LangGraph 的 checkpointer（检查点）：
- 传入 checkpointer 后，Agent 会把每个 thread_id 的对话状态持久化保存
- 同一个 thread_id 多次 invoke，模型就能"看到"之前的消息，实现多轮记忆
- 不同 thread_id 之间互相隔离，互不干扰
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from chat_model import build_chat_model

# 检查点存储：InMemorySaver 存在内存里（进程退出即丢失）。
# 生产环境可换成 SqliteSaver / PostgresSaver 等持久化后端，用法完全一致。
checkpointer = InMemorySaver()

agent = create_agent(
    model=build_chat_model(),
    tools=[],  # 这里只演示记忆，不需要工具
    system_prompt="你是一个简洁的中文助手，记住用户在对话中提供的信息。",
    checkpointer=checkpointer,
)


def chat(text: str, thread_id: str):
    """在指定会话线程里发一句话，返回 Agent 回复。"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


def main() -> None:
    thread = "user-1"

    print("第 1 轮:")
    reply = chat("你好，我叫泽鹏，我最喜欢的运动是打篮球。", thread)
    print("  Agent:", reply)

    print("第 2 轮（换个话题）:")
    reply = chat("帮我推荐一部科幻电影。", thread)
    print("  Agent:", reply)

    # 关键：这一轮没有任何新信息，但 Agent 能答对，说明它记住了前几轮
    print("第 3 轮（考察记忆）:")
    reply = chat("我叫什么名字？我喜欢什么运动？", thread)
    print("  Agent:", reply)

    # 换一个 thread_id，记忆是隔离的，Agent 不应该知道泽鹏
    print("\n另一个会话（thread=user-2，记忆隔离）:")
    reply = chat("我叫什么名字？", "user-2")
    print("  Agent:", reply)


if __name__ == "__main__":
    main()
