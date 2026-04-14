"""
Веб-сервер для RAG системы по дискретной математике.
Запускает API на порту 5000 и отдаёт index.html.
"""
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_session import Session
import sqlite3
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from retrieval import PDFVectorIndexer
    from generator import Generator
    from tester import TestingModule, Question
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все модули лежат в этой же папке.")
    sys.exit(1)

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = 'rag-dm-secret-key-2024'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
CORS(app, supports_credentials=True)

indexer = None
generator = None
tester = None

# Хранилище активных тестовых сессий (в памяти)
active_sessions = {}

def build_context(results: list) -> str:
    context_parts = []
    for page_num, chunk_idx, chunk_text, score in results:
        context_parts.append(f"[Страница {page_num}]\n{chunk_text}")
    return "\n\n".join(context_parts).strip()

def init_rag():
    global indexer, generator, tester
    
    print("=" * 60)
    print("Инициализация RAG системы...")
    print("=" * 60)
    
    pdf_path = "DM2024-szhatyi_774_-1551-1662.pdf"
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
            print("Это займёт 15-20 минут при первом запуске.")
            indexer.process()
        else:
            print(f"База данных найдена. Чанков: {count}")
    else:
        print("База данных не найдена. Начинаю индексацию PDF...")
        indexer.process()

    print("Загрузка LLM модели...")
    generator = Generator(model_path=model_path)
    
    print("Инициализация модуля тестирования...")
    tester = TestingModule(
        indexer=indexer,
        generator=generator,
        db_path="testing.db",
        chunks_per_question=10,
        questions_per_session=5
    )
    
    print("=" * 60)
    print("Сервер готов к работе!")
    print("Откройте браузер и перейдите по адресу: http://127.0.0.1:5000")
    print("=" * 60)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"answer": "Вопрос пустой."}), 400
    
    print(f"\n[Консультация] Вопрос: {question}")
    
    try:
        results = indexer.search_similar(question, top_k=3)
        
        if not results:
            return jsonify({"answer": "В учебнике не найдено релевантных фрагментов."})
        
        context = build_context(results)
        answer = generator.generate(context, question)
        
        print(f"[Консультация] Ответ: {answer[:100]}...")
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return jsonify({"answer": f"Произошла ошибка: {str(e)}"}), 500

# ========== API для тестирования ==========

@app.route('/test/start', methods=['POST'])
def start_test():
    """Начинает новую сессию тестирования, возвращает первый вопрос."""
    try:
        # Генерируем вопросы
        questions = tester.generate_questions_for_session()
        
        if not questions:
            return jsonify({"error": "Не удалось сгенерировать вопросы. Проверьте наличие чанков."}), 500
        
        # Создаём ID сессии
        session_id = str(uuid.uuid4())
        
        # Сохраняем сессию
        active_sessions[session_id] = {
            "questions": questions,
            "current_index": 0,
            "answers": [],
            "total_score": 0.0
        }
        
        first_q = questions[0]
        
        return jsonify({
            "session_id": session_id,
            "question": first_q.question_text,
            "question_number": 1,
            "total_questions": len(questions),
            "completed": False
        })
        
    except Exception as e:
        print(f"Ошибка старта теста: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test/answer', methods=['POST'])
def submit_answer():
    """Принимает ответ, оценивает, возвращает следующий вопрос или результат."""
    data = request.json
    session_id = data.get('session_id')
    answer = data.get('answer', '').strip()
    
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "Сессия не найдена. Начните тест заново."}), 400
    
    sess = active_sessions[session_id]
    current_idx = sess["current_index"]
    questions = sess["questions"]
    
    if current_idx >= len(questions):
        return jsonify({"error": "Тест уже завершён."}), 400
    
    current_q = questions[current_idx]
    
    # Оцениваем ответ
    if answer.lower() == "skip":
        score = 0.0
        feedback = "Вопрос пропущен."
    else:
        score, feedback = tester.evaluate_answer(current_q, answer)
    
    sess["total_score"] += score
    sess["answers"].append({
        "question": current_q.question_text,
        "student_answer": answer,
        "expected": current_q.expected_answer,
        "score": score,
        "feedback": feedback
    })
    
    # Переходим к следующему вопросу
    sess["current_index"] += 1
    next_idx = sess["current_index"]
    
    if next_idx >= len(questions):
        # Тест завершён
        result = {
            "completed": True,
            "total_score": sess["total_score"],
            "max_score": len(questions),
            "percentage": round(sess["total_score"] / len(questions) * 100, 1),
            "feedback": feedback,
            "expected": current_q.expected_answer
        }
        # Сохраняем в БД
        tester._save_session_to_db(sess["answers"], len(questions), sess["total_score"])
        # Удаляем сессию
        del active_sessions[session_id]
        return jsonify(result)
    else:
        # Следующий вопрос
        next_q = questions[next_idx]
        return jsonify({
            "completed": False,
            "question": next_q.question_text,
            "question_number": next_idx + 1,
            "total_questions": len(questions),
            "feedback": feedback,
            "score": score,
            "expected": current_q.expected_answer
        })

# Добавим метод в TestingModule для сохранения (если нет)
def _save_session_to_db(self, answers, total_questions, total_score):
    import json
    with sqlite3.connect(self.db_path) as conn:
        conn.execute(
            """INSERT INTO test_sessions
               (total_questions, total_score, max_possible_score, details)
               VALUES (?, ?, ?, ?)""",
            (total_questions, total_score, float(total_questions), json.dumps(answers, ensure_ascii=False))
        )

# Привязываем метод к объекту tester после инициализации
TestingModule._save_session_to_db = _save_session_to_db

if __name__ == '__main__':
    init_rag()
    app.run(host='127.0.0.1', port=5000, debug=False)