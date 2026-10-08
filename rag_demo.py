"""RAG 最小示例：本地 embedding（fastembed）+ 内存向量库 + 检索增强生成。

流程：
1. 准备知识文档 -> 切分成小块（chunk）
2. 用本地 embedding 模型把每块向量化，存入向量库
3. 用户提问 -> 向量检索出最相关的几块
4. 把检索到的内容塞进提示词，交给大模型生成"有据可依"的回答

embedding 用 fastembed 的 multilingual MiniLM，完全本地运行，无需任何 API Key。
"""

from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chat_model import build_chat_model

# 模型缓存放在项目内，避免系统临时目录被清理后重复下载（约 500MB）
CACHE_DIR = str(Path(__file__).parent / ".models")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedder(Embeddings):
    """把 fastembed 适配成 LangChain 的 Embeddings 接口。"""

    def __init__(self, model_name: str, cache_dir: str):
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, list(self._model.embed([text]))[0]))


def build_vectorstore() -> InMemoryVectorStore:
    """把示例知识文档切分、向量化，灌入内存向量库。"""
    raw_docs = [
        ("公司考勤制度", "公司实行弹性工作制，核心工作时间为上午10点到下午4点。"
                        "每位员工每月有3次迟到豁免机会，超过次数每次扣款50元。"),
        ("公司休假制度", "正式员工入职满一年后享有每年15天带薪年假，"
                        "病假需提供医院证明，婚假3天，产假按国家法定标准执行。"),
        ("报销流程", "差旅费报销需在返回后7个工作日内提交，附上发票原件和审批单，"
                    "经直属主管和财务审核后，款项在15个工作日内到账。"),
        ("技术栈说明", "后端主要使用 Python 和 FastAPI，数据库使用 MySQL 和 ClickHouse，"
                      "前端使用 React，部署在 Kubernetes 集群上。"),
    ]
    documents = [
        Document(page_content=content, metadata={"source": title})
        for title, content in raw_docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "，", ""],  # 中文友好的切分边界
    )
    chunks = splitter.split_documents(documents)

    embedder = FastEmbedder(MODEL_NAME, CACHE_DIR)
    store = InMemoryVectorStore(embedder)
    store.add_documents(chunks)
    return store


def answer(question: str, vectorstore: InMemoryVectorStore) -> None:
    """检索 + 生成：先找出相关片段，再让模型基于片段回答。"""
    # 1. 检索最相关的 2 个片段
    retrieved = vectorstore.similarity_search(question, k=2)

    print(f"\n问：{question}")
    print("检索到的上下文：")
    for doc in retrieved:
        print(f"  - [{doc.metadata['source']}] {doc.page_content}")

    # 2. 把检索结果拼进提示词（RAG 的 "A" = Augmented）
    context = "\n".join(doc.page_content for doc in retrieved)
    prompt = (
        f"请仅根据以下资料回答问题，资料中没有的信息就明确说'资料中未提及'，不要编造。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )

    # 3. 生成回答
    response = build_chat_model().invoke(prompt)
    print("回答：", response.content)


def main() -> None:
    print("正在构建向量库（首次会加载本地 embedding 模型）...")
    vectorstore = build_vectorstore()

    answer("年假有几天？", vectorstore)
    answer("迟到了会怎样？", vectorstore)
    answer("公司用什么编程语言写后端？", vectorstore)
    answer("食堂菜单有什么？", vectorstore)  # 资料里没有，考察模型是否老实说"未提及"


if __name__ == "__main__":
    main()
