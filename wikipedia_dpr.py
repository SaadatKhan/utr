import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import requests
import gzip
import pickle
import os

class FullWikipediaRetriever:
    def __init__(self, cache_dir="./wiki_cache"):
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.passages = []
        self.index = None
    
    def download_wikipedia_dpr_dump(self):
        """Download the full Wikipedia DPR passages file"""
        print("Downloading Wikipedia DPR passages...")
        
        cache_file = os.path.join(self.cache_dir, "wikipedia_passages.pkl")
        
        if os.path.exists(cache_file):
            print("Loading cached Wikipedia passages...")
            with open(cache_file, 'rb') as f:
                self.passages = pickle.load(f)
            print(f"Loaded {len(self.passages)} cached passages")
            return
        
        # Download the TSV file with all Wikipedia passages
        url = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
        
        print("Downloading Wikipedia passages (this may take a while - ~9GB)...")
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            # Save compressed file
            gz_file = os.path.join(self.cache_dir, "psgs_w100.tsv.gz")
            
            with open(gz_file, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192)):
                    f.write(chunk)
            
            print("Extracting and processing passages...")
            
            # Read and process the TSV
            with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.passages = []
            for i, line in enumerate(tqdm(lines[1:], desc="Processing passages")):  # Skip header
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    passage_id = parts[0]
                    text = parts[1]
                    title = parts[2] if len(parts) > 2 else ""
                    
                    self.passages.append({
                        'id': passage_id,
                        'title': title,
                        'text': text
                    })
                
                # Limit for memory (remove this line for full dataset)
                if i > 100000:  # First 100k passages for testing
                    break
            
            # Cache processed passages
            with open(cache_file, 'wb') as f:
                pickle.dump(self.passages, f)
            
            # Clean up
            os.remove(gz_file)
            
            print(f"Processed {len(self.passages)} Wikipedia passages")
        else:
            print("Failed to download Wikipedia data")
            return False
        
        return True
    
    def create_embeddings(self, batch_size=64, max_passages=None):
        """Create embeddings for all passages"""
        if not self.passages:
            print("No passages loaded!")
            return None
        
        # Limit passages if specified
        passages_to_process = self.passages[:max_passages] if max_passages else self.passages
        
        embeddings_file = os.path.join(self.cache_dir, f"embeddings_{len(passages_to_process)}.npy")
        
        if os.path.exists(embeddings_file):
            print("Loading cached embeddings...")
            return np.load(embeddings_file)
        
        print(f"Creating embeddings for {len(passages_to_process)} passages...")
        
        texts = []
        for passage in passages_to_process:
            # Combine title and text
            combined = f"{passage['title']}: {passage['text']}" if passage['title'] else passage['text']
            texts.append(combined)
        
        # Create embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Cache embeddings
        np.save(embeddings_file, embeddings)
        
        return embeddings
    
    def build_index(self, embeddings):
        """Build FAISS index"""
        print("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} passages")
    
    def search(self, query, top_k=10):
        """Search for any query in Wikipedia"""
        if self.index is None:
            raise ValueError("Index not built!")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.passages):
                passage = self.passages[idx]
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'id': passage['id'],
                    'title': passage['title'],
                    'text': passage['text']
                })
        
        return results
    
    def setup(self, max_passages=50000):
        """Setup complete Wikipedia retrieval system"""
        print("Setting up full Wikipedia retrieval system...")
        
        # Download/load Wikipedia data
        if not self.download_wikipedia_dpr_dump():
            return False
        
        # Limit passages for memory management
        if max_passages and len(self.passages) > max_passages:
            print(f"Limiting to first {max_passages} passages for memory management")
            self.passages = self.passages[:max_passages]
        
        # Create embeddings
        embeddings = self.create_embeddings()
        if embeddings is None:
            return False
        
        # Build index
        self.build_index(embeddings)
        
        print("Full Wikipedia retrieval system ready!")
        return True

# Usage
retriever = FullWikipediaRetriever()

# Setup (this will download ~9GB initially, but then cache everything)
if retriever.setup(max_passages=50000):  # Start with 50k passages
    
    # Now you can search for ANYTHING
    test_queries = [
        "quantum entanglement explained",
        "who discovered penicillin",
        "how do rocket engines work",
        "what is the capital of Mongolia",
        "Byzantine Empire history",
        "photosynthesis chemical equation",
        "machine learning algorithms",
        "solar system formation",
        "DNA structure discovery",
        "ancient Greek philosophy"
    ]
    
    print("\n" + "="*60)
    print("UNIVERSAL WIKIPEDIA SEARCH")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        results = retriever.search(query, top_k=3)
        
        for result in results:
            print(f"Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"Title: {result['title']}")
            print(f"Text: {result['text'][:200]}...")
            print()