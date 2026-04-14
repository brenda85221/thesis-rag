import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

MAX_HISTORY = 6   # 保留最近幾輪對話（問＋答各算一輪）


# ── PDF 載入 ───
def load_pdf_chunks(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"論文共 {len(pages)} 頁")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "．", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    print(f"切成 {len(chunks)} 個 chunks")
    return chunks


# ── 向量索引 ──
def get_vectorstore(chunks, index_dir="thesis_index"):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(index_dir):
        print("偵測到既有索引，載入中...")
        vectorstore = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        print("索引載入完成。")
    else:
        print("未找到索引，建立中（第一次需要約 1-2 分鐘）...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_dir)
        print(f"索引已儲存到：{index_dir}")
    return vectorstore


# ── Query Rewriting：把含指代的問題展開成獨立查詢 ───
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一個查詢改寫助理。
根據「對話歷史」與「使用者最新問題」，
將最新問題改寫成一個**完整、獨立、不含指代**的學術檢索查詢（繁體中文）。

規則：
- 若問題已經完整清楚，直接回傳原問題即可，不要過度展開。
- 若問題含有「它」「這個」「A1」「剛才說的」「再詳細」「第一點」等指代，
  請根據對話歷史補全，讓改寫後的查詢不需要上下文也能理解。
- 只輸出改寫後的查詢字串，不要加任何說明或引號。
""",
    ),
    ("human", "對話歷史：\n{history}\n\n使用者最新問題：{question}"),
])


def rewrite_query(llm: ChatOpenAI, history: list, question: str) -> str:
    """用 LLM 把含指代的問題展開成獨立查詢。"""
    if not history:
        return question   # 無歷史時直接用原問題

    # 把歷史格式化成純文字
    history_text = ""
    for msg in history:
        if isinstance(msg, HumanMessage):
            history_text += f"使用者：{msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"助理：{msg.content}\n"

    chain = REWRITE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"history": history_text, "question": question})
    return rewritten.strip()


# ── 主要 QA Prompt（含對話歷史）───
QA_PROMPT = PromptTemplate.from_template("""
你是一位學術論文問答助理，只能根據提供的論文內容回答問題。
如果論文中沒有提到相關資訊，請明確回答「論文中未提及」，不要自行捏造。

【對話歷史（供參考上下文）】
{history}

【論文相關段落】
{context}

【當前問題】
{question}

請用繁體中文、條列或短段落回答，必要時引用論文中的數值或專有名詞。
""")


def format_history(history: list) -> str:
    """將 LangChain message 物件轉成純文字歷史。"""
    if not history:
        return "（無）"
    lines = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            lines.append(f"使用者：{msg.content}")
        elif isinstance(msg, AIMessage):
            # 回答可能很長，只保留前 300 字避免 prompt 過大
            content = msg.content[:300] + "…" if len(msg.content) > 300 else msg.content
            lines.append(f"助理：{content}")
    return "\n".join(lines)


# ── 互動主迴圈 ──────
def interactive_loop(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    history: list = []   # 存放 HumanMessage / AIMessage

    print("\n=== 論文問答系統（含對話記憶）===")
    print("可直接用「它」「第一點」「再詳細說明」等方式追問，系統會自動理解上下文")
    print("輸入 quit 離開\n")

    demo_questions = [
        "這篇論文的研究目的是什麼？",
        "使用了哪些資料來源？",
        "資料品質管控（QC）的步驟有哪些？",
        "主要的研究結論是什麼？",
    ]
    print("示範問題：")
    for i, q in enumerate(demo_questions, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        raw_query = input("你的問題：").strip()
        if not raw_query:
            continue
        if raw_query.lower() in ["quit", "exit", "q"]:
            print("結束。")
            break

        # ① Query Rewriting：把含指代的問題展開
        rewritten = rewrite_query(llm, history, raw_query)
        if rewritten != raw_query:
            print(f"  ↳ 查詢改寫為：「{rewritten}」")

        # ② 向量檢索（用改寫後的查詢）
        source_docs = retriever.invoke(rewritten)
        context = "\n\n".join(doc.page_content for doc in source_docs)

        # ③ 組合 Prompt 並呼叫 LLM
        history_text = format_history(history)
        prompt_value = QA_PROMPT.format(
            history=history_text,
            context=context,
            question=raw_query,         # 回答時仍顯示用戶原始問題
        )
        answer = llm.invoke(prompt_value).content

        print(f"\n答案：\n{answer}")

        # ④ 顯示來源頁碼
        pages = sorted({doc.metadata.get("page", 0) + 1 for doc in source_docs})
        print(f"\n參考來源頁碼：第 {pages} 頁\n")
        print("-" * 60)

        # ⑤ 更新對話歷史（保留最近 MAX_HISTORY 輪）
        history.append(HumanMessage(content=raw_query))
        history.append(AIMessage(content=answer))
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]


# ── 入口 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("請在 .env 中設定 OPENAI_API_KEY")

    pdf_path = "thesis.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到論文檔案：{pdf_path}")

    chunks = load_pdf_chunks(pdf_path)
    vectorstore = get_vectorstore(chunks)
    interactive_loop(vectorstore)