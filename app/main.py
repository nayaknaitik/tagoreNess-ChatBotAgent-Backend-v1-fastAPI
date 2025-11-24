from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from app.enhanced_rag import chat

app = FastAPI(title="Tagore Wisdom API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class Reference(BaseModel):
    work: str
    category: str
    year: str
    excerpt: str

class ChatResponse(BaseModel):
    reply: str
    references: List[Reference]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest):
    try:
        result = chat(body.session_id, body.message)
        return ChatResponse(
            reply=result["reply"],
            references=result["references"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "message": "Tagore Wisdom API is running"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
