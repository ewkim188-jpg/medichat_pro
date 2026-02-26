from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pyngrok
from pyngrok import ngrok
import nest_asyncio
import sys

# 기존 RAG 모델 로드
from model import load_llm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

app = FastAPI(title="MediChat Pro Backend API")

db_faiss = None
llm = None
from model import ManualRAG, DB_FAISS_PATH

class QueryRequest(BaseModel):
    query: str
    mode: str = "Patient"
    chat_history: list = []

@app.on_event("startup")
async def startup_event():
    global db_faiss, llm
    print("Loading Embeddings and Vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    if os.path.exists(DB_FAISS_PATH):
        db_faiss = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector DB loaded successfully.")
    else:
        print(f"Warning: Vector DB not found at {DB_FAISS_PATH}")
    
    print("Loading LLM (Llama-2, this will take some memory)...")
    llm = load_llm()
    if llm is None:
        print("Error: Could not load LLM. API will fail.")
    else:
        print("LLM loaded successfully.")

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    if not llm or not db_faiss:
        raise HTTPException(status_code=500, detail="Model or DB not loaded.")
    
    rag_chain = ManualRAG(llm, db_faiss)
    result = rag_chain({'query': req.query, 'mode': req.mode})
    
    # Extract source documents for debugging/logging, exclude them from direct response if not needed
    source_docs = [doc.page_content[:200] + "..." for doc in result['source_documents']]
    
    return {
        "answer": result['result'],
        "sources": source_docs
    }

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ngrok":
        nest_asyncio.apply()
        # You might need an auth token for ngrok if you face limits, but usually it works without for random URLs.
        public_url = ngrok.connect(8000).public_url
        print("\n" + "="*50)
        print(f"Public API URL: {public_url}")
        print("COPY THIS URL AND SET IT AS 'API_URL' IN STREAMLIT CLOUD SECRETS!")
        print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
