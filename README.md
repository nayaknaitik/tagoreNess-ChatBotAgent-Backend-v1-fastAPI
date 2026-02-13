Tagore Wisdom Chatbot -
An AI-powered conversational assistant that delivers emotionally-aware, citation-backed wisdom from Rabindranath Tagore's complete works. Perfect for life guidance, philosophical insights, and personal growth.

✨ Key Features
🧠 Emotion-Aware Retrieval - Analyzes your emotional state & problem type

📚 Complete Tagore Corpus - 6,444+ chunks from 11 major works (Gitanjali, Sadhana, etc.)

🎯 Multi-Dimensional Search - Semantic + metadata filtering (domain, tone, genre)

📖 Structured Citations - Every response includes work name, year, & excerpt

💬 Conversation Memory - Remembers your chat history per session

🎨 Beautiful UI - Next.js + TailwindCSS with smooth animations

⚡ FastAPI Backend - Production-ready with Groq Llama-3.3 inference

🏗️ Architecture
text
Frontend (Next.js) ↔ FastAPI API ↔ RAG Engine (LangChain + FAISS)
                                    ↓
                       Context Analyzer → Multi-Dim Retrieval → LLM
                                    ↓
                         Groq Llama-3.3 (llama-3.3-70b-versatile)
🚀 Quick Start
Prerequisites
Node.js 18+

Python 3.12+

Groq API Key (Free tier works!)

Backend Setup
bash
# Clone & navigate
git clone <your-repo>
cd ragBackend

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt

# Copy env template
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Start backend
uvicorn app.main:app --reload
Frontend Setup
bash
cd frontend  # or your Next.js app directory
npm install
npm run dev
Visit: http://localhost:3000

📋 Example Usage
text
User: "I feel overwhelmed by work"
Bot: "I understand work pressure can feel heavy...

"Work is not man's mission. Man's mission is to serve..." 
- From Sadhana (1913)

**Practical Steps:**
1. Take 5 deep breaths before starting work
2. Set one small goal for today only
3. End your day with 10 minutes of reflection

What small step feels possible right now?"

References:
• Sadhana (1913, essays)
• Gitanjali (1912, poetry)
🛠️ Tech Stack
Layer	Technology
Frontend	Next.js 14, React 18, TypeScript, TailwindCSS, Framer Motion
Backend	FastAPI, Python 3.12, UV Package Manager
RAG Pipeline	LangChain 0.2.x, FAISS Vector Store, HuggingFace Embeddings
Embeddings	sentence-transformers/all-MiniLM-L6-v2 (384-dim)
LLM	Groq Llama-3.3-70b-versatile (~200ms latency)
Database	In-memory session store (Redis-ready)
🔍 Smart Retrieval System
Context Analysis - Classifies emotion, problem domain, urgency

Semantic Search - Finds top 12 similar passages

Multi-Dimensional Scoring - Ranks by domain match + tone + genre

Top-4 Selection - Best passages fed to LLM with citations

text
Score = 0.4×semantic + 0.3×domain + 0.2×tone + 0.1×genre_bonus
📂 Project Structure
text
ragBackend/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   └── enhanced_rag.py      # Core RAG logic
├── data_collection/         # Book download & processing
├── processed_data/          # Vector store + JSON corpus
├── requirements.txt
└── .env.example

frontend/
├── app/
│   └── chat/
│       └── page.tsx         # Main chat page
├── components/              # ChatBubble, Header, Input
└── store/                   # Zustand user state
⚙️ Environment Variables
Create .env from .env.example:

text
GROQ_API_KEY=your_groq_api_key_here
🧪 API Endpoints
Endpoint	Method	Description
/api/health	GET	Health check
/api/chat	POST	Chat with citations
Request:

json
{
  "session_id": "user-123",
  "message": "How to find inner peace?"
}
Response:

json
{
  "reply": "Full response with Tagore wisdom...",
  "references": [
    {"work": "Gitanjali", "year": "1912", "category": "poetry", "excerpt": "..."}
  ]
}
🔄 Development Workflow
bash
# Backend (Terminal 1)
uvicorn app.main:app --reload

# Frontend (Terminal 2)  
npm run dev

# Process new books (if needed)
uv run python data_collection/process_books.py
📈 Performance
Vector Store: 6,444 documents (~1M words)

Query Latency: <500ms end-to-end

Embedding Speed: 10k sentences/sec (CPU)

LLM Latency: ~200ms (Groq)

🎯 Use Cases
Personal growth & life guidance

Philosophical discussions

Emotional support with literary grounding

Educational tool for Tagore studies

Demo for RAG/multi-modal retrieval research

🤝 Contributing
Fork the repo

Create feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push & PR!

See CONTRIBUTING.md for details.

📄 License
This project is MIT licensed - use freely!

🙏 Acknowledgments
Rabindranath Tagore - Source of infinite wisdom

Groq - Lightning-fast inference

LangChain - RAG framework

Project Gutenberg - Free Tagore texts

HuggingFace - Open embeddings

📞 Support
🤔 Found a bug? Open Issue

💡 Feature request? Discussion

🆘 Need help? Check Issues

<div align="center"> <sub>Built with ❤️ for wisdom seekers everywhere. <br> "Where the mind is without fear..." — Rabindranath Tagore</sub> </div>
⭐ Star us on GitHub if this helps you!
