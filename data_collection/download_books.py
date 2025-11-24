import requests
import json
import time
from pathlib import Path

class TagoreDataCollector:
    def __init__(self, output_dir="raw_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Project Gutenberg book IDs for Tagore
        self.gutenberg_books = {
            6686: {'title': 'Gitanjali', 'year': 1912, 'category': 'poetry'},
            7164: {'title': 'The Gardener', 'year': 1913, 'category': 'poetry'},
            6523: {'title': 'Fruit Gathering', 'year': 1916, 'category': 'poetry'},
            6606: {'title': 'Stray Birds', 'year': 1916, 'category': 'poetry'},
            4117: {'title': 'The Crescent Moon', 'year': 1913, 'category': 'poetry'},
            7166: {'title': 'Sadhana', 'year': 1913, 'category': 'essays'},
            40766: {'title': 'Nationalism', 'year': 1917, 'category': 'essays'},
            45134: {'title': 'Creative Unity', 'year': 1922, 'category': 'essays'},
            8688: {'title': 'Personality', 'year': 1917, 'category': 'essays'},
            7001: {'title': 'The Home and the World', 'year': 1916, 'category': 'novels'},
            7666: {'title': 'Chitra', 'year': 1913, 'category': 'plays'},
        }
    
    def _clean_gutenberg_text(self, raw_text):
        """Remove Gutenberg header/footer"""
        start_markers = ["*** START OF", "***START OF"]
        end_markers = ["*** END OF", "***END OF"]
        
        start_pos = -1
        for marker in start_markers:
            pos = raw_text.find(marker)
            if pos != -1:
                start_pos = raw_text.find('\n', pos) + 1
                break
        
        end_pos = len(raw_text)
        for marker in end_markers:
            pos = raw_text.find(marker)
            if pos != -1:
                end_pos = pos
                break
        
        if start_pos > 0 and end_pos > start_pos:
            return raw_text[start_pos:end_pos].strip()
        
        return raw_text.strip()
    
    def download_from_gutenberg(self, book_id, metadata):
        """Download a single book from Project Gutenberg"""
        urls = [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        ]
        
        for url in urls:
            try:
                print(f"  Trying: {url}")
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    text = self._clean_gutenberg_text(response.text)
                    
                    if len(text) > 1000:  # Ensure we got actual content
                        print(f"  ✓ Downloaded successfully ({len(text)} chars)")
                        return {
                            'work': metadata['title'],
                            'content': text,
                            'year': metadata['year'],
                            'category': metadata['category'],
                            'source': 'project_gutenberg',
                            'source_id': book_id,
                            'source_url': url
                        }
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        return None
    
    def download_all_books(self):
        """Download all Tagore books from Project Gutenberg"""
        print("=" * 60)
        print("Starting Tagore Books Download")
        print("=" * 60)
        
        all_books = []
        failed_books = []
        
        for book_id, metadata in self.gutenberg_books.items():
            print(f"\n[{len(all_books)+1}/{len(self.gutenberg_books)}] Downloading: {metadata['title']}")
            
            book_data = self.download_from_gutenberg(book_id, metadata)
            
            if book_data:
                all_books.append(book_data)
                
                # Save individual book
                filename = f"{book_id}_{metadata['title'].replace(' ', '_')}.txt"
                filepath = self.output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(book_data['content'])
                print(f"  ✓ Saved to: {filepath}")
            else:
                failed_books.append(metadata['title'])
                print(f"  ✗ Failed to download")
            
            time.sleep(2)  # Be respectful to servers
        
        # Save complete corpus as JSON
        corpus_file = self.output_dir / "tagore_complete_corpus.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_books': len(all_books),
                    'failed_books': failed_books,
                    'collection_date': time.strftime('%Y-%m-%d')
                },
                'books': all_books
            }, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✓ Download Complete!")
        print(f"  Total books downloaded: {len(all_books)}")
        print(f"  Failed downloads: {len(failed_books)}")
        print(f"  Corpus saved to: {corpus_file}")
        print("=" * 60)
        
        return all_books

# Run this script directly
if __name__ == "__main__":
    collector = TagoreDataCollector()
    books = collector.download_all_books()
