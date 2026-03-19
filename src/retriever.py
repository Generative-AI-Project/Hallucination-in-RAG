import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load BOTH files to ensure "Super Bowl 50" is included
files = ['train-v1.1.json', 'dev-v1.1.json']
raw_contexts = []

print("🔄 Loading and merging datasets...")
for file in files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            for article in data['data']:
                for paragraph in article['paragraphs']:
                    raw_contexts.append(paragraph['context'])
    except FileNotFoundError:
        print(f"⚠️ Warning: {file} not found. Skipping.")

# 2. Deduplicate and Chunk
unique_contexts = list(set(raw_contexts))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

final_chunks = []
for ctx in unique_contexts:
    final_chunks.extend(text_splitter.split_text(ctx))

# 3. FRESH Vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Generating embeddings for {len(final_chunks)} chunks...")
embeddings = model.encode(final_chunks, show_progress_bar=True)

# 4. RESET and Build FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 5. Save files
faiss.write_index(index, "squad_index.faiss")
with open("squad_chunks.pkl", "wb") as f:
    pickle.dump(final_chunks, f)

print(f"✅ DONE! Index Size: {index.ntotal} | Chunk List Size: {len(final_chunks)}")

def final_verify(query):
    D, I = index.search(model.encode([query]), k=1)
    print(f"\nTop Result for '{query}':")
    print(final_chunks[I[0][0]])

final_verify("Which NFL team won Super Bowl 50?")