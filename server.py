from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_session import Session
import sqlite3
import sys
import os
import uuid
import hashlib
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from retrieval import PDFVectorIndexer
    from tester import TestingModule
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = 'rag-dm-secret-key-2024'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
CORS(app, supports_credentials=True)

indexer = None
tester = None

answer_cache = {}
CACHE_MAX_SIZE = 200

# ---------- Функции для извлечения лучшего предложения с весами ----------
def split_into_sentences(text: str) -> list:
    """Разбивает текст на предложения по . ! ? и переводу строки."""
    sentences = re.split(r'(?<=[.!?])\s+|[.!?]$', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def get_definition_bonus(sentence: str, question: str) -> float:
    """
    Вычисляет дополнительный вес предложения, если оно похоже на определение.
    Бонус начисляется за:
    - наличие маркеров определения: 'называется', 'это', 'определяется как', 'означает', 'есть'
    - начало предложения с термина из вопроса (первые 3-5 слов)
    """
    bonus = 0.0
    lower_sent = sentence.lower()
    lower_q = question.lower()

    # 1. Маркеры определения
    definition_markers = ['называется', 'это', 'определяется как', 'означает', 'есть', 'является']
    for marker in definition_markers:
        if marker in lower_sent:
            bonus += 0.15
            break

    # 2. Начало предложения содержит ключевые слова вопроса (первые 2-3 слова)
    # Из вопроса извлекаем первые 2-3 значимых слова (игнорируем "что", "как", "такое")
    stopwords = {'что', 'как', 'такое', 'такой', 'зачем', 'почему', 'где', 'когда'}
    q_words = [w for w in lower_q.split() if w not in stopwords]
    if q_words:
        # Берём первые 2-3 значимых слова
        key_phrase = ' '.join(q_words[:3])
        if lower_sent.startswith(key_phrase) or lower_sent.find(key_phrase) < 20:
            bonus += 0.2

    # 3. Предложение начинается с глагола в третьем лице (признак определения)
    if re.match(r'^[А-Яа-я]+(уется|ется|ляет|вляется)', lower_sent):
        bonus += 0.1

    return bonus

def build_answer_from_results(results, question, indexer) -> str:
    """
    Ищет среди всех чанков и их предложений самое релевантное,
    добавляя бонус за определения. Возвращает лучшее предложение.
    """
    if not results:
        return "Не найдено релевантных фрагментов."

    q_emb = indexer.model.encode(question, convert_to_numpy=True).astype(np.float32)

    best_data = {"score": -1.0, "sentence": "", "page": None}

    for page_num, chunk_idx, chunk_text, _ in results:
        sentences = split_into_sentences(chunk_text)
        if not sentences:
            continue

        for sent in sentences:
            sent_emb = indexer.model.encode(sent, convert_to_numpy=True).astype(np.float32)
            cosine = np.dot(q_emb, sent_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(sent_emb))
            bonus = get_definition_bonus(sent, question)
            total_score = cosine + bonus

            if total_score > best_data["score"]:
                best_data["score"] = total_score
                best_data["sentence"] = sent.strip()
                best_data["page"] = page_num

    if best_data["sentence"] and best_data["score"] > 0.4:
        return f"[Стр. {best_data['page']}] {best_data['sentence']}"
    else:
        # fallback: первый чанк целиком
        first = results[0]
        return f"[Стр. {first[0]}]\n{first[2][:500]}..."

# ---------- Инициализация ----------
def init_rag():
    global indexer, tester
    print("=" * 60)
    print("Инициализация RAG системы (с приоритетом определений)...")
    print("=" * 60)

    pdf_path = "DM2024-szhatyi_774_-1551-1662.pdf"
    db_path = "ontologer.db"

    if not os.path.exists(pdf_path):
        print(f"Файл {pdf_path} не найден!")
        sys.exit(1)

    print("Инициализация индексатора с FAISS...")
    indexer = PDFVectorIndexer(
        pdf_path=pdf_path,
        db_path=db_path,
        resume=True,
        model_name="intfloat/multilingual-e5-large",
        use_faiss=True,
        faiss_index_path="faiss.index"
    )

    if os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM chunk_embeddings")
            count = cur.fetchone()[0]
        if count == 0:
            print("База пуста. Индексация PDF... (5-10 минут)")
            indexer.process()
        else:
            print(f"База найдена. Чанков: {count}")
    else:
        print("Индексация...")
        indexer.process()

    print("Инициализация модуля тестирования (без LLM)...")
    tester = TestingModule(
        indexer=indexer,
        db_path="testing.db",
        questions_per_session=5,
        similarity_threshold=0.8
    )

    print("=" * 60)
    print("Сервер готов! http://127.0.0.1:5000")
    print("=" * 60)

# ---------- Эндпоинт консультации ----------
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"answer": "Вопрос пустой."}), 400

    cache_key = hashlib.md5(question.lower().encode('utf-8')).hexdigest()
    if cache_key in answer_cache:
        print(f"[Кэш] {question[:50]}...")
        return jsonify({"answer": answer_cache[cache_key]})

    print(f"\n[Консультация] {question}")

    try:
        results = indexer.search_similar(question, top_k=5)
        if not results:
            answer = "Не найдено релевантных фрагментов."
        else:
            answer = build_answer_from_results(results, question, indexer)

        if len(answer_cache) >= CACHE_MAX_SIZE:
            answer_cache.pop(next(iter(answer_cache)))
        answer_cache[cache_key] = answer
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Ошибка: {e}")
        return jsonify({"answer": f"Ошибка: {str(e)}"}), 500

# ---------- Тестирование (без изменений) ----------
active_sessions = {}

@app.route('/test/start', methods=['POST'])
def start_test():
    try:
        questions = tester.generate_questions_for_session()
        if not questions:
            return jsonify({"error": "Нет вопросов в пуле."}), 500
        session_id = str(uuid.uuid4())
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
        print(f"Ошибка: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test/answer', methods=['POST'])
def submit_answer():
    data = request.json
    session_id = data.get('session_id')
    answer = data.get('answer', '').strip()
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "Сессия не найдена."}), 400
    sess = active_sessions[session_id]
    current_idx = sess["current_index"]
    questions = sess["questions"]
    if current_idx >= len(questions):
        return jsonify({"error": "Тест завершён."}), 400
    current_q = questions[current_idx]
    if answer.lower() == "skip":
        score = 0.0
        feedback = "Пропущено."
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
    sess["current_index"] += 1
    next_idx = sess["current_index"]
    if next_idx >= len(questions):
        import json
        with sqlite3.connect(tester.db_path) as conn:
            conn.execute(
                "INSERT INTO test_sessions (total_questions, total_score, max_possible_score, details) VALUES (?, ?, ?, ?)",
                (len(questions), sess["total_score"], float(len(questions)), json.dumps(sess["answers"], ensure_ascii=False))
            )
        result = {
            "completed": True,
            "total_score": sess["total_score"],
            "max_score": len(questions),
            "percentage": round(sess["total_score"] / len(questions) * 100, 1),
            "feedback": feedback,
            "expected": current_q.expected_answer
        }
        del active_sessions[session_id]
        return jsonify(result)
    else:
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


if __name__ == '__main__':
    init_rag()
    app.run(host='127.0.0.1', port=5000, debug=False)
