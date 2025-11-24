import json
from pathlib import Path
from typing import List, Dict
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TagoreDocumentProcessor:
    def __init__(self, raw_data_dir="raw_data", output_dir="processed_data"):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Problem domain keywords
        self.problem_domains = {
            'relationships': ['love', 'family', 'friendship', 'marriage', 'companion', 'bond', 'intimacy'],
            'work_life': ['work', 'duty', 'labor', 'service', 'occupation', 'toil', 'effort'],
            'spirituality': ['god', 'divine', 'prayer', 'soul', 'spirit', 'meditation', 'worship'],
            'creativity': ['art', 'music', 'poetry', 'beauty', 'creation', 'expression', 'imagination'],
            'freedom': ['freedom', 'liberty', 'independence', 'liberation', 'constraint', 'bondage'],
            'education': ['learning', 'teaching', 'knowledge', 'wisdom', 'child', 'student', 'education'],
            'nature': ['nature', 'sky', 'earth', 'tree', 'flower', 'river', 'season', 'bird'],
            'suffering': ['pain', 'sorrow', 'grief', 'suffering', 'loss', 'death', 'tears'],
            'joy': ['joy', 'happiness', 'delight', 'bliss', 'laughter', 'pleasure', 'gladness'],
            'truth': ['truth', 'honesty', 'sincerity', 'reality', 'genuine', 'authentic']
        }
        
        # Emotional tones
        self.emotional_tones = {
            'comforting': ['peace', 'calm', 'gentle', 'rest', 'comfort', 'solace', 'quiet'],
            'inspiring': ['courage', 'strength', 'hope', 'rise', 'awaken', 'light', 'dawn'],
            'philosophical': ['truth', 'meaning', 'purpose', 'understanding', 'wisdom', 'reflect'],
            'challenging': ['question', 'doubt', 'struggle', 'confront', 'fight', 'resist']
        }
    
    def load_corpus(self):
        """Load the downloaded corpus"""
        corpus_file = self.raw_data_dir / "tagore_complete_corpus.json"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data['books']
    
    def _extract_problem_domains(self, text):
        """Extract relevant problem domains from text"""
        text_lower = text.lower()
        found_domains = []
        
        for domain, keywords in self.problem_domains.items():
            if any(keyword in text_lower for keyword in keywords):
                found_domains.append(domain)
        
        return found_domains if found_domains else ['general']
    
    def _analyze_emotional_tone(self, text):
        """Analyze emotional tone of text"""
        text_lower = text.lower()
        tone_scores = {}
        
        for tone, keywords in self.emotional_tones.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            tone_scores[tone] = score
        
        if any(tone_scores.values()):
            return max(tone_scores, key=tone_scores.get)
        return 'neutral'
    
    def _assess_complexity(self, text):
        """Assess text complexity based on word length"""
        words = text.split()
        if not words:
            return 'simple'
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        if avg_word_length < 4.5:
            return 'simple'
        elif avg_word_length < 6:
            return 'moderate'
        else:
            return 'complex'
    
    def _identify_form(self, text):
        """Identify if poetry or prose"""
        lines = text.split('\n')
        if not lines:
            return 'prose'
        
        short_lines = sum(1 for line in lines if len(line.strip()) < 60)
        
        return 'poetry' if short_lines / len(lines) > 0.4 else 'prose'
    
    def process_books(self, chunk_size=500, chunk_overlap=100):
        """Process books into enriched chunks"""
        print("=" * 60)
        print("Processing Tagore Books")
        print("=" * 60)
        
        books = self.load_corpus()
        all_documents = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
        )
        
        for i, book in enumerate(books):
            print(f"\n[{i+1}/{len(books)}] Processing: {book['work']}")
            
            # Split into chunks
            chunks = text_splitter.split_text(book['content'])
            print(f"  Created {len(chunks)} chunks")
            
            for chunk_idx, chunk in enumerate(chunks):
                # Rich metadata
                metadata = {
                    # Source info
                    'work': book['work'],
                    'category': book['category'],
                    'year': book['year'],
                    'source': book['source'],
                    'source_url': book.get('source_url', ''),
                    
                    # Chunk info
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    
                    # Content analysis
                    'problem_domains': self._extract_problem_domains(chunk),
                    'emotional_tone': self._analyze_emotional_tone(chunk),
                    'complexity_level': self._assess_complexity(chunk),
                    'literary_form': self._identify_form(chunk),
                    
                    # For filtering
                    'word_count': len(chunk.split()),
                    'char_count': len(chunk)
                }
                
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                all_documents.append(doc)
        
        print(f"\n✓ Total processed chunks: {len(all_documents)}")
        
        # Save processed documents
        output_file = self.output_dir / "processed_documents.json"
        
        # Convert Documents to serializable format
        serializable_docs = [
            {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in all_documents
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_documents': len(serializable_docs),
                'documents': serializable_docs
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved to: {output_file}")
        print("=" * 60)
        
        return all_documents

# Run this script
if __name__ == "__main__":
    processor = TagoreDocumentProcessor()
    documents = processor.process_books()
