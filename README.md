# Thesis RAG Q&A System
RAG 架構，將學術論文 PDF 轉為可查詢的 AI 問答系統。

##  系統架構
1. PDF 讀取 ： `PyPDFLoader` 逐頁擷取文字
2. 切塊 ： `RecursiveCharacterTextSplitter`（chunk_size=600, overlap=80）
3. 向量化 ： `OpenAI text-embedding-ada-002` Embedding
4. 索引儲存 ： `FAISS` 本地向量索引
5. 檢索 ： cosine similarity，取 top-4 相關段落
6. 回答生成：`GPT-4o-mini` + Custom Prompt，回傳答案與來源頁碼

## 對話記憶（新增）
支援上下文連續追問，無需重複說明前文脈絡。

運作流程：
使用者輸入（可含指代）-> Query Rewriting — LLM 將問題展開為完整查詢 -> 向量檢索（使用改寫後的查詢）-> LLM 回答（同時參考：對話歷史 + 論文段落）-> 更新對話歷史

## 如何開始?

1. 安裝套件與設定相關環境
pip install -r requirements.txt

2. 設定 API Key到.env
OPENAI_API_KEY=sk-你的key

3. 放入 PDF(thesis.pdf)至同層資料夾，執行
python thesis_qa.py

4. 開始你的問答^^

## 實際操作範例
<img width="697" height="352" alt="image" src="https://github.com/user-attachments/assets/149b9bbf-61a0-4b1d-9cf8-859db149215a" />
<img width="658" height="174" alt="image" src="https://github.com/user-attachments/assets/3ff4fef7-ad01-4e1d-99ee-023d5b948729" />
<img width="734" height="305" alt="image" src="https://github.com/user-attachments/assets/1eaabfbb-ea61-4f05-90ce-fbcb894d95c5" />

### 對話記憶（新增）
<img width="641" height="421" alt="螢幕擷取畫面 2026-04-14 142838" src="https://github.com/user-attachments/assets/13f3daba-492e-450c-b64f-c2190a64279e" />


