"""文件操作 Agent 最小实现：read / write / edit / list 四个工具 + 沙箱限制。

和 agent_demo.py 结构完全相同（create_agent + 工具 + 提示词），
重点看两件事：
1. 工具就是普通 Python 函数，模型只负责决定"调哪个、传什么参数"
2. 所有路径都强制限制在 workspace/ 目录内（模拟 harness 的沙箱）
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from chat_model import build_chat_model

# 沙箱根目录：Agent 只能操作这里面的文件
WORKSPACE = Path(__file__).parent / "workspace"


def _safe_path(path: str) -> Path:
    """路径校验：解析后必须仍在 workspace 内，否则拒绝。"""
    resolved = (WORKSPACE / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE.resolve())):
        raise ValueError(f"越权访问：{path} 不在 workspace/ 内")
    return resolved


@tool
def list_files() -> str:
    """列出 workspace 目录下的所有文件。"""
    files = [f"'{f.name}' ({f.stat().st_size} 字节)"
             for f in sorted(WORKSPACE.iterdir()) if f.is_file()]
    return "\n".join(files) if files else "workspace 是空的"


@tool
def read_file(path: str) -> str:
    """读取 workspace 内某个文件的完整内容。path 是相对 workspace 的文件名，如 'config.txt'。"""
    try:
        return _safe_path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"错误：文件不存在 {path}"
    except ValueError as exc:
        return f"错误：{exc}"


@tool
def write_file(path: str, content: str) -> str:
    """在 workspace 内创建或覆盖一个文件并写入内容。"""
    p = _safe_path(path)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字符）"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """把 workspace 内文件中唯一的 old_text 替换为 new_text。old_text 必须在文件中恰好出现一次，否则报错。"""
    p = _safe_path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        return f"错误：'{old_text}' 在 {path} 中出现了 {count} 次（需要恰好 1 次），请提供更长的上下文使其唯一"
    p.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return f"修改成功：'{old_text}' -> '{new_text}'"


SYSTEM_PROMPT = (
    "你是一个文件操作助手，只能操作 workspace 目录内的文件。"
    "修改文件前必须先用 read_file 查看内容，修改后用 read_file 验证结果。"
    "按步骤调用工具完成任务，最后简要总结你做了什么。"
)

agent = create_agent(
    model=build_chat_model(),
    tools=[list_files, read_file, write_file, edit_file],
    system_prompt=SYSTEM_PROMPT,
)


def setup_workspace() -> None:
    """每次运行前重置 workspace，保证 demo 可重复。"""
    WORKSPACE.mkdir(exist_ok=True)
    (WORKSPACE / "config.txt").write_text(
        "app_name = learnAgent-demo\n"
        "host = 127.0.0.1\n"
        "port = 8000\n"
        "debug = false\n",
        encoding="utf-8",
    )
    (WORKSPACE / "todo.md").write_text(
        "# 待办\n\n- [ ] 把服务端口改为 8080\n- [x] 跑通 ChatModel\n",
        encoding="utf-8",
    )


def main() -> None:
    setup_workspace()

    result = agent.invoke(
        {"messages": [{"role": "user",
                       "content": "帮我把 config.txt 里的端口改成 8080，"
                                  "并把 todo.md 里对应的待办项打上勾，"
                                  "最后写一个冒泡排序golang代码txt文件。"}]}
    )

    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()