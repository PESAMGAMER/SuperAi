import os
import json
import faiss
import torch
import numpy as np
import requests
import time
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template, jsonify

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # ตรวจสอบชื่อให้ตรงกับที่ติดตั้งจริง
FOLDER_PATHS = ["data", "family"]  # แก้ไขเป็น 3 folders
INDEX_PATH = "faiss.index"
TEXTS_PATH = "texts.json"

# ตรวจสอบ GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ ใช้งานบน: {device.upper()}")

app = Flask(__name__)
embedder = SentenceTransformer("intfloat/multilingual-e5-base")
embedder.to(device)  # ย้าย model ไปที่ GPU

# ตัวแปร global
texts = []
index = None

def load_laws(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = []
    # ถ้าเป็น list
    if isinstance(data, list):
        for item in data:
            try:
                # กรณีที่มี law_name
                if 'law_name' in item and 'section_num' in item and 'section_content' in item:
                    results.append(f"{item['law_name']} มาตรา {item['section_num']}:\n{item['section_content']}")
                # กรณีที่มีแค่ section_content
                elif 'section_content' in item:
                    results.append(item['section_content'])
            except Exception as e:
                print(f"  ⚠️ ข้าม record ที่ไม่สมบูรณ์: {str(e)}")
                continue
    # ถ้าเป็น dict หรือรูปแบบอื่น
    else:
        print(f"  ⚠️ ไม่รองรับ format ของไฟล์: {filepath}")
        return []
        
    return results

def load_laws_from_folders(folder_paths: List[str]) -> List[str]:
    all_texts = []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"✗ ไม่พบโฟลเดอร์ {folder_path}")
            continue
            
        print(f"📚 โหลดไฟล์กฎหมายจากโฟลเดอร์ {folder_path}")
        
        # นับจำนวน files ทั้งหมดก่อน
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        print(f"📑 พบไฟล์ .json ทั้งหมด {len(json_files)} ไฟล์")
        
        # ใช้ tqdm สำหรับแสดง progress bar
        for filename in tqdm(json_files, desc=f"โหลดไฟล์จาก {folder_path}"):
            file_path = os.path.join(folder_path, filename)
            try:
                texts = load_laws(file_path)
                all_texts.extend(texts)
                print(f"  ✓ โหลด {filename} สำเร็จ ({len(texts)} มาตรา)")
            except Exception as e:
                print(f"  ✗ ไม่สามารถโหลด {filename}: {str(e)}")
                
        print(f"✅ โหลดจาก {folder_path} เสร็จสิ้น รวม {len(all_texts)} มาตรา\n")
        
    print(f"📊 สรุป: โหลดข้อมูลทั้งหมด {len(all_texts)} มาตรา จาก {len(folder_paths)} โฟลเดอร์")
    return all_texts

def embed_chunks(texts: List[str], batch_size: int = 16) -> np.ndarray:
    print("🔍 สร้าง Embeddings...")
    return embedder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True,
                           device=device)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    print("📦 สร้าง FAISS Index...")
    if device == "cuda":
        # ใช้ GPU สำหรับ FAISS
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_similar_contexts(query: str, texts: List[str], index: faiss.IndexFlatIP, top_k: int = 5) -> List[str]:
    print("🧠 ค้นหาข้อมูล...")
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True, device=device)
    scores, indices = index.search(query_embedding, top_k)

    threshold = 0.3  # ลด threshold เล็กน้อย
    filtered_results = [(texts[i], scores[0][idx]) for idx, i in enumerate(indices[0])
                        if scores[0][idx] > threshold]

    if not filtered_results:
        # fallback ถ้าไม่มีข้อมูลผ่าน threshold
        print("⚠️ ไม่พบข้อมูลที่ผ่าน threshold ใช้ top-k แทน")
        filtered_results = [(texts[i], scores[0][idx]) for idx, i in enumerate(indices[0])]

    filtered_results.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in filtered_results]

def generate_answer_ollama(context: str, query: str, max_tokens: int = 512) -> str:
    print("🔍 วิเคราะห์บริบทกฎหมายที่เกี่ยวข้อง...")
    
    if context.strip():
        # ถ้ามีบริบทกฎหมายที่เกี่ยวข้อง
        print(f"📖 พบข้อกฎหมายที่เกี่ยวข้อง {len(context.split('มาตรา'))-1} มาตรา")
        prompt = f"""คุณคือผู้เชี่ยวชาญด้านกฎหมายที่สามารถอธิบายข้อกฎหมายให้เข้าใจง่าย

บริบทกฎหมายที่เกี่ยวข้อง:
{context}

คำถาม: {query}

กรุณาตอบโดย:
1. อธิบายด้วยภาษาที่เข้าใจง่าย
2. อ้างอิงมาตราที่เกี่ยวข้อง
3. ยกตัวอย่างประกอบ (ถ้ามี)"""

    else:
        # ถ้าไม่พบบริบทกฎหมายที่เกี่ยวข้อง
        print("⚠️ ไม่พบข้อกฎหมายที่เกี่ยวข้องโดยตรง")
        prompt = f"""คุณคือผู้เชี่ยวชาญด้านกฎหมายที่ช่วยตอบคำถามทั่วไป

คำถาม: {query}

กรุณาตอบโดย:
1. ใช้ภาษาที่เข้าใจง่าย
2. อธิบายตามหลักกฎหมายทั่วไป
3. แนะนำให้ปรึกษาผู้เชี่ยวชาญถ้าต้องการคำแนะนำเฉพาะเจาะจง"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        })

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"❌ ติดต่อ Ollama ไม่สำเร็จ: {response.status_code}"

    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()

    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': '❌ โปรดระบุคำถาม'}), 400

    top_contexts = search_similar_contexts(question, texts, index)
    context = "\n\n".join(top_contexts)
    answer = generate_answer_ollama(context, question)

    return jsonify({
        'context': context,
        'answer': answer,
        'time_taken': f"{time.time() - start_time:.2f} วินาที"
    })

if __name__ == "__main__":
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        print("📂 โหลดข้อมูลจากแคช...")
        index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, "r", encoding="utf-8") as f:
            texts = json.load(f)
    else:
        texts = load_laws_from_folders(FOLDER_PATHS)
        if not texts:
            print("❌ ไม่พบข้อมูลกฎหมายในโฟลเดอร์")
            exit(1)

        print(f"\n📝 พบข้อมูลกฎหมายทั้งหมด {len(texts)} มาตรา")
        embeddings = embed_chunks(texts)
        index = build_faiss_index(embeddings)

        faiss.write_index(index, INDEX_PATH)
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

    print("\n🎯 ระบบ RAG (Ollama) พร้อมใช้งาน!\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
