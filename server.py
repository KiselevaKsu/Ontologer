"""
Веб-сервер для RAG системы по дискретной математике.
Запускает API на порту 5000 и отдаёт index.html.
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from retrieval import PDFVectorIndexer
    from generator import Generator
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что retrieval.py и generator.py лежат в этой же папке.")
    sys.exit(1)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

indexer = None
generator = None

def build_context(results: list) -> str:
    """Формирует контекст из результатов поиска."""
    context_parts = []
    for page_num, chunk_idx, chunk_text, score in results:
        context_parts.append(f"[Страница {page_num}]\n{chunk_text}")
    return "\n\n".join(context_parts).strip()

def init_rag():
    """Инициализация индексатора и генератора."""
    global indexer, generator
    
    print("=" * 60)
    print("Инициализация RAG системы...")
    print("=" * 60)
    
    pdf_path = "DM2024-сжатый.pdf"
    db_path = "ontologer.db"
    model_path = "T-lite-it-2.1-Q4_K_M.gguf"

    if not os.path.exists(pdf_path):
        print(f"Файл {pdf_path} не найден!")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Файл {model_path} не найден!")
        sys.exit(1)

    print("Инициализация индексатора...")
    indexer = PDFVectorIndexer(
        pdf_path=pdf_path,
        db_path=db_path,
        resume=True,
        model_name="intfloat/multilingual-e5-large"
    )

    if os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM chunk_embeddings")
            count = cur.fetchone()[0]
        if count == 0:
            print("База данных пуста. Начинаю индексацию PDF...")
            print(" Это займёт 15-20 минут при первом запуске.")
            indexer.process()
        else:
            print(f"База данных найдена. Чанков: {count}")
    else:
        print("База данных не найдена. Начинаю индексацию PDF...")
        indexer.process()

    print("Загрузка LLM модели...")
    generator = Generator(model_path=model_path)
    
    print("=" * 60)
    print("Сервер готов к работе!")
    print("Откройте браузер и перейдите по адресу: http://127.0.0.1:5000")
    print("=" * 60)

@app.route('/')
def index():
    """Отдаёт index.html при открытии корневого URL."""
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """API эндпоинт для вопросов."""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"answer": "Вопрос пустой."}), 400
    
    print(f"\nВопрос: {question}")
    
    try:
        results = indexer.search_similar(question, top_k=3)
        
        if not results:
            return jsonify({"answer": "В учебнике не найдено релевантных фрагментов."})
        
        context = build_context(results)
        
        print("Генерация ответа...")
        answer = generator.generate(context, question)
        
        print(f"Ответ: {answer[:100]}...")
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return jsonify({"answer": f"Произошла ошибка: {str(e)}"}), 500

if __name__ == '__main__':
    init_rag()
    app.run(host='127.0.0.1', port=5000, debug=False)