

import numpy as np
from sentence_transformers import SentenceTransformer
from config import MODEL_PATH_EMBEDDING, CHUNK_SIZE, MAX_AEM_SIZE

class AEM:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_PATH_EMBEDDING)
        self.aem_memory = []

    def update_aem(self, new_chunks, similarities):
        # Add new chunks with similarities
        for chunk, sim in zip(new_chunks, similarities):
            self.aem_memory.append({'text': chunk, 'similarity': sim})
        
        # Sort and keep only top by similarity
        self.aem_memory = sorted(self.aem_memory, key=lambda x: x['similarity'], reverse=True)[:MAX_AEM_SIZE]

    def get_aem(self):
        return [item['text'] for item in self.aem_memory]

    def simple_chunk_and_search(self, text, query):
        # Split into chunks
        chunks = [text[i:i+CHUNK_SIZE].strip() for i in range(0, len(text), CHUNK_SIZE) if len(text[i:i+CHUNK_SIZE].strip()) > 20]
        if not chunks:
            return []

        print(f"Created {len(chunks)} chunks")

        # Embed all
        all_texts = [query] + chunks
        embeddings = self.model.encode(all_texts, normalize_embeddings=True)

        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        # Cosine similarity (dot product since they're normalized)
        similarities = np.dot(chunk_embs, query_emb)

        self.update_aem(chunks, similarities)

        # Sort results
        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked_indices:
            if similarities[idx] > 0:  # basic threshold
                results.append({
                    'rank': len(results) + 1,
                    'similarity': round(float(similarities[idx]), 3),
                    'text': chunks[idx]
                })
        return results
