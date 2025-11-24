import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document


# ================================
# Load environment variables
# ================================
load_dotenv()
HF_API = os.getenv("hugging_face_api")
GROQ_API = os.getenv("GROQ_API_KEY")


# ================================
# Load PDF and create RAG pipeline
# ================================
PDF_PATH = "app/tagore.pdf"  # <---- Change this to your actual PDF name

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file '{PDF_PATH}' not found.")

# Load PDF
loader = PyPDFLoader(PDF_PATH)
corpus = loader.load()

#load Documents
with open("app/tagore.json" , "r") as f :
    data = json.load(f)
texts = []
for item in data :
    for concept in item['content']:
        chunk = f"{item['title']}: {concept}"
        texts.append(chunk)

docs = [Document(page_content = t) for t in texts]


# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
final_documents = text_splitter.split_documents(corpus)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector DB
vectorstore = FAISS.from_documents(final_documents, embeddings)

vectorstore.add_documents(docs)

retriever = vectorstore.as_retriever()

# LLM (Groq)
llm = ChatGroq(model_name="openai/gpt-oss-120b", groq_api_key=GROQ_API)


# ================================
# Prompts
# ================================
contextualize_q_system_prompt = (
    "You are a direct, practical counselor who uses ideas from Tagore’s writings strictly as "
    "principles for reasoning — not as poetry, not as metaphors, and not as literary imitation. "
    "Your job is to help the user solve real-life problems with clear logic, concise reasoning, "
    "and specific, actionable steps. Avoid vague comfort, spiritual talk, moral lectures, or flowery language. "

    "Use the retrieved context only when it directly strengthens your explanation or provides a "
    "useful principle. If it does not help, ignore it completely. Do not stretch or force a link "
    "between the context and the user's problem. Never rewrite the context or turn it poetic. "

    "Your tone must stay human, calm, and rational — like a grounded friend who tells the truth "
    "without sugarcoating. Be straightforward, challenge faulty assumptions, point out contradictions, "
    "and highlight what actually matters. Prioritize clarity over inspiration, and reasoning over emotion. "

    "Focus on three things: (1) what is really going on, (2) why it matters, and (3) what the user can "
    "realistically do next. Avoid generalities. Avoid philosophy. Avoid metaphors. Stick to practical insight."
)


contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# Answer prompt
answer_q_system_prompt = (
    "You are an empathetic assistant inspired by the human values and emotional clarity found in Tagore's writings. "
    "However, you do NOT imitate his poetic style, do NOT use metaphors, and do NOT produce long philosophical passages. "
    "Your job is to help the user with realistic, grounded, emotionally-aware guidance. "
    "Use a calm, simple, reflective tone that feels human and supportive—never flowery, dramatic, or abstract. "

    "You are provided with structured thematic insights (JSON) drawn from Tagore-like principles such as self-understanding, "
    "discipline, simplicity, balance, fear, relationships, and resilience. Use these themes ONLY when they clearly apply to "
    "the user’s question. Do not force a theme or quote; select only what is relevant and restate it in your own simple words. "
    "Never invent new Tagore ideas or pretend a passage is written by Tagore. "

    "Always Include one quotation from Tagore's work that is relevant to the user's question. "
    "Your response must always follow this structure:\n"
    "1) **Understanding** — Briefly restate the user's problem to show clarity.\n"
    "2) **Relevant Insight** — Select 1–2 applicable ideas from the retrieved context or JSON themes. Paraphrase them simply, without poetic or philosophical tones.\n"
    "3) **Practical Guidance** — Give 2–4 actionable steps the user can apply right now.\n"
    "4) **Grounding** — End with a short, calm, encouraging note without metaphors or flowery language.\n"

    "Never reveal the internal JSON data, vector store content, retrieved chunks, or any part of the PDF. "
    "Never mention retrieval, embeddings, or RAG processing. "
    "Even if the user asks directly, politely provide a helpful explanation without exposing the internal data or system structure."


    "If retrieved context is weak, irrelevant, or unhelpful, ignore it. Always prioritize accuracy, emotional steadiness, and practical usefulness. "
    "Your goal is to help the user think clearly, act wisely, and stay emotionally balanced."
)


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_q_system_prompt + "\n\nRelevant Text:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# ================================
# RAG Chain with Retriever
# ================================
def retrieve_documents(state: dict):
    """Retrieve documents from the vector store based on input and chat history"""
    retrieved_docs = history_aware_retriever.invoke({
        "input": state.get("input", ""),
        "chat_history": state.get("chat_history", [])
    })
    # Convert documents to formatted string
    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    else:
        context = "No relevant context found in the document."
    return context


rag_chain = (
    {
        "input": lambda x: x["input"],
        "context": lambda x: retrieve_documents(x),  # NOW ACTUALLY RETRIEVES!
        "chat_history": lambda x: x.get("chat_history", [])
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)


# ================================
# Memory system
# ================================
SESSION_STORE = {}


def get_session_history(session_id):
    """Get or create chat history for a session"""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)


# ================================
# Chat function
# ================================
def chat(session_id: str, message: str):
    """Process a chat message and return the response"""
    return conversational_rag_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )


# ================================
# FASTAPI APP
# ================================
app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat_api(data: ChatRequest):
    """API endpoint for chat"""
    response = chat(data.session_id, data.message)
    return {"reply": response}


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}


# ================================
# RUN SERVER
# ================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)