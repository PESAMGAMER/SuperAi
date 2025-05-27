# ระบบถาม-ตอบกฎหมายไทย (Thai Law QA System)

ระบบตอบคำถามเกี่ยวกับกฎหมายไทยโดยใช้ RAG (Retrieval-Augmented Generation) และ Ollama

## การติดตั้ง

1. ติดตั้ง Python 3.8 หรือใหม่กว่า
2. ติดตั้ง Ollama จาก https://ollama.ai
3. ติดตั้ง Ollama Model:
```bash
   ollama pull llama3.2
```

4. สร้าง Virtual Environment (ถ้ายังไม่สร้าง)
```bash
   create virtual environment python -m venv venv
```

5. เข้าใช้งาน venv
```bash
   venv\Scripts\activate
```   

6. ติดตั้ง Dependencies:
```bash
   pip install -r requirements.txt
```

## การเตรียมข้อมูล

1. สร้างโฟลเดอร์ `data` ในโปรเจค
2. นำไฟล์ JSON ที่มีข้อมูลกฎหมายไปไว้ในโฟลเดอร์ `data`
   - รูปแบบไฟล์ JSON ต้องประกอบด้วย: law_name, section_num, section_content

## การรันระบบ

1. รันแอปพลิเคชัน:
```bash
python thai_law_rag.py
```

2. เปิดเว็บบราวเซอร์ไปที่ http://localhost:5000

## วิธีใช้งาน

1. พิมพ์คำถามเกี่ยวกับกฎหมายในช่องคำถาม
2. กดปุ่ม "ส่งคำถาม" หรือกด Enter
3. รอระบบประมวลผลและแสดงคำตอบ
4. สามารถเลือกคำถามตัวอย่างได้จากปุ่มด้านล่าง

## กระบวนการทำงาน

1. **การเตรียมข้อมูล**
   - โหลดข้อมูลกฎหมายจากไฟล์ JSON
   - สร้าง Embeddings ด้วย SentenceTransformer
   - สร้าง FAISS Index สำหรับการค้นหา

2. **การประมวลผลคำถาม**
   - รับคำถามจากผู้ใช้
   - ค้นหาบริบทกฎหมายที่เกี่ยวข้องด้วย FAISS
   - ส่งคำถามและบริบทไปยัง Ollama
   - แสดงคำตอบที่ได้แก่ผู้ใช้

3. **การแคช**
   - ระบบจะสร้างไฟล์แคช (faiss.index และ texts.json)
   - ใช้ไฟล์แคชเพื่อลดเวลาในการโหลดครั้งต่อไป
