import os
import json
import requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

origins = ["http://localhost:5173", "https://wmaszyna.netlify.app/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables loaded at startup
chunks = []
base_chunk = {}
system_prompt = ""
index = None
chunk_texts = []

API_KEY = os.getenv("GEMINI_API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

class ChatRequest(BaseModel):
    history: list  # list of {"role": "user"/"assistant", "message": "..."}

def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

def load_faiss_index(index_path, emb_path):
    index = faiss.read_index(index_path)
    embeddings = np.load(emb_path)
    return index, embeddings

def get_top_chunks(query, index, texts, model, top_k=2):
    query_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_emb)
    _, indices = index.search(query_emb, top_k)
    return [texts[i] for i in indices[0]]

def construct_prompt(query, system_prompt, history_text, relevant_chunks, base_chunk):
    prompt = f"{system_prompt}\n{history_text}\nContext:\n"
    for chunk in relevant_chunks:
        prompt += f"- ({chunk['source']}) {chunk['content']}\n"
    prompt += f"- ({base_chunk['source']}) {base_chunk['content']}\n"
    return prompt

def get_answer(prompt):
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return None

@app.on_event("startup")
def startup_event():
    global chunks, base_chunk, system_prompt, index, chunk_texts

    chunks = load_jsonl("output_chunks.jsonl")
    config = load_jsonl("rag_prompt_config.jsonl")[0]
    base_chunk = config['base_chunk']
    system_prompt = config['system_prompt']

    chunk_texts = [chunk['content'] for chunk in chunks]

    # Precomputed index and embeddings
    if os.path.exists("faiss_index.index") and os.path.exists("chunk_embeddings.npy"):
        index, _ = load_faiss_index("faiss_index.index", "chunk_embeddings.npy")
    else:
        embeddings = model.encode(chunk_texts, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, "faiss_index.index")
        np.save("chunk_embeddings.npy", embeddings)

@app.post("/chat")
def chat(req: ChatRequest):
    history_text = ""
    query = ""
    for item in req.history:
        if item["role"] == "user":
            history_text += f"User query: {item['message']}\n"
            query = item["message"]
        elif item["role"] == "assistant":
            history_text += f"Gemini response: {item['message']}\n"

    relevant_chunks = get_top_chunks(query, index, chunks, model)
    prompt = construct_prompt(query, system_prompt, history_text, relevant_chunks, base_chunk)
    answer = get_answer(prompt)
    if answer is None:
        return {"error": "Failed to get a response from Gemini"}
    return {"response": answer}
