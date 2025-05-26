import os
import json
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from fastapi.middleware.cors import CORSMiddleware

# Initialize app
app = FastAPI()

origins = ["http://localhost:5173", "https://wmaszyna.netlify.app/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data once at startup
chunks = None
base_chunk = None
system_prompt = None
API_KEY = os.getenv("GEMINI_API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def load_chunks(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

def retrieve_relevant_chunks(query, chunks, model, top_k=3):
    texts = [chunk['content'] for chunk in chunks]
    chunk_embeddings = model.encode(texts, convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(chunk_embeddings)

    embedding_dim = chunk_embeddings.shape[1]
    index_file = "faiss_index.index"
    if not os.path.exists(index_file):
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(chunk_embeddings)
        faiss.write_index(index, index_file)
    else:
        index = faiss.read_index(index_file)

    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def construct_prompt(query, chunks, model, base_chunk, system_prompt, history):
    relevant_chunks = retrieve_relevant_chunks(query, chunks, model)
    prompt = f"{system_prompt}\n{history}\nContext:\n"
    for chunk in relevant_chunks:
        prompt += f"- ({chunk['source']}) {chunk['content']}\n"
    prompt += f"- ({base_chunk['source']}) {base_chunk['content']}\n"
    return prompt

def get_answer(prompt):
    prompt_data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    response = requests.post(
        URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(prompt_data)
    )

    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return None

class ChatRequest(BaseModel):
    history: list  # list of {"role": "user"/"assistant", "message": "..."}

@app.on_event("startup")
def load_configs():
    global chunks, base_chunk, system_prompt
    chunks = load_chunks("output_chunks.jsonl")
    config = load_chunks("rag_prompt_config.jsonl")[0]
    base_chunk = config['base_chunk']
    system_prompt = config['system_prompt']

@app.post("/chat")
def chat(req: ChatRequest):
    history_text = ""
    for item in req.history:
        if item["role"] == "user":
            history_text += f"User query: {item['message']}\n"
            query = item["message"]
        elif item["role"] == "assistant":
            history_text += f"Gemini response: {item['message']}\n"

    prompt = construct_prompt(query, chunks, model, base_chunk, system_prompt, history_text)
    answer = get_answer(prompt)
    if answer is None:
        return {"error": "Failed to get a response from Gemini"}
    return {"response": answer}
