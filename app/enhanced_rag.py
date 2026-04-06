import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment
load_dotenv()
GROQ_API = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated model
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

FALLBACK_RESPONSE = "I cannot provide a grounded answer based on the available sources."


class ContextAwareRetriever:
    """Advanced retriever with context analysis"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def analyze_user_context(self, query: str, age: int = 25) -> Dict:
        """Analyze user query for context"""
        analysis_llm = ChatGroq(
            model_name=GROQ_MODEL, groq_api_key=GROQ_API, temperature=0.3
        )

        analysis_prompt = ChatPromptTemplate.from_template(
            """Analyze this user query and extract context:

User query: {query}
User age: {age}

Return ONLY a valid JSON object with these exact keys:
{{"emotion": "anxious/sad/confused/angry/hopeful/neutral", "category": "relationships/work_life/spirituality/creativity/freedom/suffering/joy", "urgency": "low/medium/high", "tone": "comforting/inspiring/philosophical/challenging"}}

JSON:"""
        )

        chain = analysis_prompt | analysis_llm | StrOutputParser()
        result = chain.invoke({"query": query, "age": age})

        try:
            import re

            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                context = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
        except:
            context = {
                "emotion": "neutral",
                "category": "general",
                "urgency": "medium",
                "tone": "philosophical",
            }

        return context

    def multi_dimensional_retrieve(
        self, query: str, user_context: Dict, k: int = 4
    ) -> List[Document]:
        """Retrieve with multi-dimensional weighted scoring"""
        base_docs = self.vectorstore.similarity_search(query, k=k * 3)
        max_semantic = k * 3

        scored_docs = []
        for rank, doc in enumerate(base_docs):
            metadata = doc.metadata

            semantic_score = 1.0 - (rank / max_semantic)
            domain_match = (
                1.0
                if user_context.get("category") in metadata.get("problem_domains", [])
                else 0.0
            )
            emotional_alignment = (
                1.0
                if metadata.get("emotional_tone") == user_context.get("tone")
                else 0.0
            )

            metadata_bonus = 0.0
            category = user_context.get("category")
            doc_category = metadata.get("category", "")
            if category == "relationships" and doc_category == "poetry":
                metadata_bonus = 1.0
            elif category == "spirituality" and doc_category == "essays":
                metadata_bonus = 1.0

            final_score = (
                0.4 * semantic_score
                + 0.3 * domain_match
                + 0.2 * emotional_alignment
                + 0.1 * metadata_bonus
            )

            if DEBUG_MODE:
                print(
                    f"  Doc {rank + 1} | semantic={semantic_score:.2f} domain={domain_match} emotion={emotional_alignment} bonus={metadata_bonus} | final={final_score:.3f}"
                )

            scored_docs.append((doc, final_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

    def validate_grounding(self, response: str, docs: List[Document]) -> bool:
        """Validate that response is grounded in retrieved documents"""
        if not docs or not response:
            return False

        # Check for at least one quoted string
        quote_pattern = r'"([^"]{10,})"'  # quotes with at least 10 chars
        quotes = re.findall(quote_pattern, response)

        if not quotes:
            if DEBUG_MODE:
                print("  [VALIDATION] No quotes found in response")
            return False

        # Check each quote exists in at least one doc
        doc_texts = [doc.page_content.lower() for doc in docs]
        doc_texts_combined = " ".join(doc_texts)

        valid_quotes = 0
        for quote in quotes:
            quote_lower = quote.lower()
            if quote_lower in doc_texts_combined:
                valid_quotes += 1
            else:
                # Fuzzy match - check if significant words overlap
                quote_words = set(quote_lower.split())
                if len(quote_words) >= 4:
                    overlap = sum(1 for w in quote_words if w in doc_texts_combined)
                    if overlap / len(quote_words) >= 0.7:
                        valid_quotes += 1

        if valid_quotes == 0:
            if DEBUG_MODE:
                print(f"  [VALIDATION] No quotes matched documents")
            return False

        # Check keyword overlap between response and docs
        response_words = set(response.lower().split())
        doc_words = set(doc_texts_combined.split())
        common_words = response_words & doc_words
        overlap_ratio = len(common_words) / max(len(response_words), 1)

        if overlap_ratio < 0.15:
            if DEBUG_MODE:
                print(f"  [VALIDATION] Low keyword overlap: {overlap_ratio:.2f}")
            return False

        if DEBUG_MODE:
            print(
                f"  [VALIDATION] Passed - {valid_quotes} quotes, {overlap_ratio:.2f} keyword overlap"
            )

        return True


class EnhancedTagoreRAG:
    """Enhanced RAG system with citations"""

    def __init__(self):
        print("Initializing Enhanced Tagore RAG System...")

        self.documents = self._load_processed_documents()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = self._create_vectorstore()

        self.llm = ChatGroq(
            model_name=GROQ_MODEL, groq_api_key=GROQ_API, temperature=0.7
        )

        self.context_retriever = ContextAwareRetriever(self.vectorstore, self.llm)
        self.session_store = {}
        self.conversational_chain = self._build_rag_chain()

        print("✓ System ready!")

    def _load_processed_documents(self) -> List[Document]:
        """Load processed documents from JSON"""
        doc_file = Path("processed_data/processed_documents.json")

        if not doc_file.exists():
            raise FileNotFoundError(
                f"Processed documents not found at {doc_file}. "
                "Please run: uv run python data_collection/process_books.py"
            )

        with open(doc_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = [
            Document(page_content=doc["content"], metadata=doc["metadata"])
            for doc in data["documents"]
        ]

        print(f"✓ Loaded {len(documents)} processed documents")
        return documents

    def _create_vectorstore(self):
        """Create FAISS vector store"""
        vectorstore_path = Path("processed_data/faiss_index")

        if vectorstore_path.exists():
            print("✓ Loading existing vector store...")
            try:
                vectorstore = FAISS.load_local(
                    str(vectorstore_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"✓ Loaded vector store with {vectorstore.index.ntotal} vectors")
                return vectorstore
            except Exception as e:
                print(f"⚠ Error loading vector store: {e}")
                print("✓ Creating new vector store...")
        else:
            print("✓ Creating new vector store...")

        # Create new vector store
        vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        vectorstore.save_local(str(vectorstore_path))
        print(f"✓ Vector store created with {vectorstore.index.ntotal} vectors")

        return vectorstore

    def _build_rag_chain(self):
        """Build the RAG chain with citations"""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given chat history and latest user question, "
                    "reformulate it as a standalone question.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            contextualize_q_prompt,
        )

        # Enhanced prompt with citation instructions
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a wise, empathetic counselor inspired by Rabindranath Tagore's values and teachings.

Context from Tagore's works:
{context}

---

## 🔒 CORE RULE: STRICTLY GROUNDED RESPONSE

* You are ONLY allowed to use the provided context.
* Do NOT introduce any ideas, advice, or knowledge not present in the context.
* Do NOT generalize beyond the retrieved passages.
* If the context is insufficient, respond with:
  "I cannot provide a grounded answer based on the available sources."

---

## 📚 CITATION ENFORCEMENT (MANDATORY)

* You MUST include 1–2 DIRECT QUOTES from the context.

* Quotes must:

  * Be EXACT (no paraphrasing)
  * Match the provided context text
  * Follow this format:

  "[Exact quote]" — From [Work Name]

* Every major insight MUST be traceable to the context.

* If you cannot find a valid quote → return fallback response.

---

## 🧠 EMOTIONAL ALIGNMENT (KEEP NATURAL)

* Acknowledge the user's emotional state with genuine empathy
* Match tone with the retrieved passages (comforting / inspiring / philosophical / challenging)
* Do NOT exaggerate emotions
* Keep responses calm, grounded, and reflective

---

## 🎯 RESPONSE STRUCTURE (UNCHANGED — FOLLOW STRICTLY)

1. Understanding

* Brief acknowledgment of user's emotional state

2. Relevant Insight

* Draw wisdom ONLY from the provided context
* Include 1–2 DIRECT QUOTES with citation

3. Practical Guidance

* Provide 2–3 actionable steps
* Each step MUST logically follow from the cited content

4. Grounding

* End with a thoughtful reflection or question

---

## 🚫 HARD RESTRICTIONS

* No hallucination
* No generic advice not supported by context
* No external knowledge
* No modern frameworks unless explicitly present in context
* No assumptions beyond retrieved text

---

## ✅ INTERNAL VALIDATION (VERY IMPORTANT)

Before generating final answer, ensure:
✔ At least one exact quote is present
✔ All insights are derived from context
✔ No external knowledge is used

If any condition fails → return fallback response.

---

## 🎯 GOAL

Produce a response that is:

* Emotionally aligned
* Fully grounded in Tagore’s writings
* Verifiable through citations
* Clear, structured, and helpful
""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # Custom retrieval with metadata preservation
        def enhanced_retrieve(state: dict):
            """Enhanced retrieval with citations"""
            user_query = state.get("input", "")

            # Analyze context
            user_context = self.context_retriever.analyze_user_context(user_query)

            if DEBUG_MODE:
                print(f"\n[DEBUG] user_context: {user_context}")

            # Multi-dimensional retrieval
            docs = self.context_retriever.multi_dimensional_retrieve(
                user_query, user_context, k=4
            )

            # Format context with source attribution
            if docs:
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    work = doc.metadata.get("work", "Unknown")
                    category = doc.metadata.get("category", "writing")
                    year = doc.metadata.get("year", "")

                    context_part = (
                        f"[Source {i}]\n"
                        f"Work: {work}\n"
                        f"Year: {year}\n"
                        f"Category: {category}\n"
                        f"Content: {doc.page_content}"
                    )
                    context_parts.append(context_part)

                # Store retrieved docs for later citation extraction
                state["retrieved_docs"] = docs
                context = "\n\n---\n\n".join(context_parts)
            else:
                state["retrieved_docs"] = []
                context = (
                    "No specific context found. Use general wisdom and principles."
                )

            return context

        # Build chain
        rag_chain = (
            {
                "input": lambda x: x["input"],
                "context": lambda x: enhanced_retrieve(x),
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | answer_prompt
            | self.llm
            | StrOutputParser()
        )

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output",
        )

        return conversational_chain

    def _get_session_history(self, session_id: str):
        """Get or create session history"""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def _extract_references(self, response: str, docs: List[Document]) -> List[Dict]:
        """Extract references from retrieved documents"""
        references = []
        seen_works = set()

        for doc in docs:
            work = doc.metadata.get("work", "Unknown Work")

            # Avoid duplicate works
            if work in seen_works:
                continue
            seen_works.add(work)

            # Create reference with all metadata
            ref = {
                "work": work,
                "category": doc.metadata.get("category", "writing"),
                "year": str(doc.metadata.get("year", "")),
                "excerpt": doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content,
            }
            references.append(ref)

        return references

    def chat(self, session_id: str, message: str) -> Dict:
        """Process chat message with safe generation loop"""

        user_context = self.context_retriever.analyze_user_context(message)
        retrieved_docs = self.context_retriever.multi_dimensional_retrieve(
            message, user_context, k=4
        )

        if DEBUG_MODE:
            print(f"[DEBUG] Retrieved {len(retrieved_docs)} docs")

        response = None
        for attempt in range(2):
            response = self.conversational_chain.invoke(
                {"input": message}, config={"configurable": {"session_id": session_id}}
            )

            if DEBUG_MODE:
                print(f"[DEBUG] Generation attempt {attempt + 1}")

            if self.context_retriever.validate_grounding(response, retrieved_docs):
                if DEBUG_MODE:
                    print("[DEBUG] Grounding validation passed")
                break
            else:
                if DEBUG_MODE:
                    print("[DEBUG] Grounding validation failed - retrying")

        if not response or not self.context_retriever.validate_grounding(
            response, retrieved_docs
        ):
            if DEBUG_MODE:
                print("[DEBUG] Using fallback response")
            response = FALLBACK_RESPONSE

        references = self._extract_references(response, retrieved_docs)

        if DEBUG_MODE:
            print(
                f"[DEBUG] Validation result: valid={self.context_retriever.validate_grounding(response, retrieved_docs)}"
            )

        return {"reply": response, "references": references}


# Initialize the RAG system (singleton pattern)
tagore_rag = EnhancedTagoreRAG()


def chat(session_id: str, message: str) -> Dict:
    """Chat function for API - returns dict with reply and references"""
    return tagore_rag.chat(session_id, message)
