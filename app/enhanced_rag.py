import os
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

class ContextAwareRetriever:
    """Advanced retriever with context analysis"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def analyze_user_context(self, query: str, age: int = 25) -> Dict:
        """Analyze user query for context"""
        analysis_llm = ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API,
            temperature=0.3
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
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                context = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
        except:
            context = {
                "emotion": "neutral",
                "category": "general",
                "urgency": "medium",
                "tone": "philosophical"
            }
        
        return context
    
    def multi_dimensional_retrieve(self, query: str, user_context: Dict, k: int = 4) -> List[Document]:
        """Retrieve with multi-dimensional filtering"""
        base_docs = self.vectorstore.similarity_search(query, k=k*3)
        
        scored_docs = []
        for doc in base_docs:
            score = 0
            metadata = doc.metadata
            
            # Match problem domain
            if user_context.get('category') in metadata.get('problem_domains', []):
                score += 3
            
            # Match emotional tone
            if metadata.get('emotional_tone') == user_context.get('tone'):
                score += 2
            
            # Prefer certain categories for specific problems
            if user_context.get('category') == 'relationships' and metadata.get('category') == 'poetry':
                score += 1
            elif user_context.get('category') == 'spirituality' and metadata.get('category') == 'essays':
                score += 1
            
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]


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
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API,
            temperature=0.7
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
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = [
            Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            )
            for doc in data['documents']
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
                    allow_dangerous_deserialization=True
                )
                print(f"✓ Loaded vector store with {vectorstore.index.ntotal} vectors")
                return vectorstore
            except Exception as e:
                print(f"⚠ Error loading vector store: {e}")
                print("✓ Creating new vector store...")
        else:
            print("✓ Creating new vector store...")
        
        # Create new vector store
        vectorstore = FAISS.from_documents(
            self.documents,
            self.embeddings
        )
        vectorstore.save_local(str(vectorstore_path))
        print(f"✓ Vector store created with {vectorstore.index.ntotal} vectors")
        
        return vectorstore
    
    def _build_rag_chain(self):
        """Build the RAG chain with citations"""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given chat history and latest user question, "
                      "reformulate it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            contextualize_q_prompt
        )
        
        # Enhanced prompt with citation instructions
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a wise, empathetic counselor inspired by Rabindranath Tagore's values and teachings.

Context from Tagore's works:
{context}

Guidelines for your response:
1. **Acknowledge** the user's emotional state with genuine empathy
2. **Draw wisdom** from Tagore's context provided above - use his actual words and ideas
3. **Include 1-2 DIRECT QUOTES** from the context (use exact text and cite the work)
4. **Provide 2-3 practical steps** the user can take immediately
5. **End with reflection** - a thoughtful question or gentle encouragement

IMPORTANT FORMAT for quotes:
"[Exact quote from context]" - From [Work Name]

Structure your response clearly with these sections:
- Understanding (brief acknowledgment)
- Relevant Insight (Tagore's wisdom with quotes)
- Practical Guidance (actionable steps)
- Grounding (reflective closing)

Be warm, clear, and grounded. Use Tagore's actual words from the context provided."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Custom retrieval with metadata preservation
        def enhanced_retrieve(state: dict):
            """Enhanced retrieval with citations"""
            user_query = state.get("input", "")
            
            # Analyze context
            user_context = self.context_retriever.analyze_user_context(user_query)
            
            # Multi-dimensional retrieval
            docs = self.context_retriever.multi_dimensional_retrieve(
                user_query,
                user_context,
                k=4
            )
            
            # Format context with source attribution
            if docs:
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    work = doc.metadata.get('work', 'Unknown')
                    category = doc.metadata.get('category', 'writing')
                    year = doc.metadata.get('year', '')
                    
                    context_part = f"[Source {i}] From '{work}' ({year}, {category}):\n{doc.page_content}"
                    context_parts.append(context_part)
                
                # Store retrieved docs for later citation extraction
                state['retrieved_docs'] = docs
                context = "\n\n---\n\n".join(context_parts)
            else:
                state['retrieved_docs'] = []
                context = "No specific context found. Use general wisdom and principles."
            
            return context
        
        # Build chain
        rag_chain = (
            {
                "input": lambda x: x["input"],
                "context": lambda x: enhanced_retrieve(x),
                "chat_history": lambda x: x.get("chat_history", [])
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
            output_messages_key="output"
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
            work = doc.metadata.get('work', 'Unknown Work')
            
            # Avoid duplicate works
            if work in seen_works:
                continue
            seen_works.add(work)
            
            # Create reference with all metadata
            ref = {
                'work': work,
                'category': doc.metadata.get('category', 'writing'),
                'year': str(doc.metadata.get('year', '')),
                'excerpt': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            references.append(ref)
        
        return references
    
    def chat(self, session_id: str, message: str) -> Dict:
        """Process chat message and return response with citations"""
        
        # Analyze and retrieve documents first
        user_context = self.context_retriever.analyze_user_context(message)
        retrieved_docs = self.context_retriever.multi_dimensional_retrieve(
            message,
            user_context,
            k=4
        )
        
        # Get response from conversational chain
        response = self.conversational_chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Extract references from retrieved documents
        references = self._extract_references(response, retrieved_docs)
        
        return {
            "reply": response,
            "references": references
        }


# Initialize the RAG system (singleton pattern)
tagore_rag = EnhancedTagoreRAG()

def chat(session_id: str, message: str) -> Dict:
    """Chat function for API - returns dict with reply and references"""
    return tagore_rag.chat(session_id, message)
