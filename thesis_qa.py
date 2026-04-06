import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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


def get_vectorstore(chunks, index_dir="thesis_index"):
    embeddings = OpenAIEmbeddings()

    if os.path.exists(index_dir):
        print("偵測到既有索引，載入中...")
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("索引載入完成。")
    else:
        print("未找到索引，建立中（第一次需要約 1-2 分鐘）...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_dir)
        print(f"索引已儲存到：{index_dir}")

    return vectorstore


def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
你是一位學術論文問答助理，只能根據提供的論文內容回答問題。
如果論文中沒有提到相關資訊，請明確回答「論文中未提及」，不要自行捏造。

論文內容：
{context}

問題：{question}

請用繁體中文、條列或短段落回答，必要時引用論文中的數值或專有名詞。
""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def interactive_loop(chain, retriever):
    print("\n=== 論文問答系統 ===")
    print("輸入問題即可查詢，輸入 quit 離開\n")

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
        query = input("你的問題：").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("結束。")
            break

        # 取得回答
        answer = chain.invoke(query)
        print(f"\n答案：\n{answer}")

        # 顯示來源頁碼
        source_docs = retriever.invoke(query)
        pages = sorted({doc.metadata.get("page", 0) + 1 for doc in source_docs})
        print(f"\n參考來源頁碼：第 {pages} 頁\n")
        print("-" * 60)


if __name__ == "__main__":
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("請在 .env 中設定 OPENAI_API_KEY")

    pdf_path = "thesis.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到論文檔案：{pdf_path}")

    chunks = load_pdf_chunks(pdf_path)
    vectorstore = get_vectorstore(chunks)
    chain, retriever = build_chain(vectorstore)
    interactive_loop(chain, retriever)