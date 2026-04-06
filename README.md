# Thesis RAG Q&A System
RAG 架構，將學術論文 PDF 轉為可查詢的 AI 問答系統。

##  系統架構
1. PDF 讀取 ： `PyPDFLoader` 逐頁擷取文字
2. 切塊 ： `RecursiveCharacterTextSplitter`（chunk_size=600, overlap=80）
3. 向量化 ： `OpenAI text-embedding-ada-002` Embedding
4. 索引儲存 ： `FAISS` 本地向量索引
5. 檢索 ： cosine similarity，取 top-4 相關段落
6. 回答生成：`GPT-4o-mini` + Custom Prompt，回傳答案與來源頁碼

## 如何開始?

1. 安裝套件與設定相關環境
pip install -r requirements.txt

2. 設定 API Key到.env
OPENAI_API_KEY=sk-你的key

3. 放入 PDF(thesis.pdf)至同層資料夾，執行
python thesis_qa.py

4. 開始你的問答^^
