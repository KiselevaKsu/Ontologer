import sqlite3
import json
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from retrieval import PDFVectorIndexer


@dataclass
class Question:
    chunk_ids: List[Tuple[int, int]]
    combined_text: str
    question_text: str
    expected_answer: str
    expected_embedding: Optional[np.ndarray] = None   # предвычисленный эмбеддинг эталона
    student_answer: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None


class TestingModule:
    def __init__(
        self,
        indexer: PDFVectorIndexer,
        db_path: str = "testing.db",
        questions_per_session: int = 5,
        similarity_threshold: float = 0.8,
        embedding_model_name: str = "intfloat/multilingual-e5-large"
    ):
        self.indexer = indexer
        self.db_path = db_path
        self.questions_per_session = questions_per_session
        self.similarity_threshold = similarity_threshold

        # Загружаем эмбеддинг-модель (она уже есть в indexer, но используем отдельно для оценки)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self._init_db()
        self._static_question_pool = self._build_question_pool()
        # Предвычисляем эмбеддинги для всех эталонных ответов
        self._precompute_expected_embeddings()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_questions INTEGER,
                    total_score REAL,
                    max_possible_score REAL,
                    details TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS question_pool (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_ids TEXT,
                    combined_text TEXT,
                    question_text TEXT,
                    expected_answer TEXT
                )
            """)

    def _build_question_pool(self) -> List[Question]:
        """Статический пул из 50 вопросов (как и раньше)."""
        pool = []
        # ----- 10.1 Фундаментальные циклы и разрезы -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое правильный разрез графа?",
            expected_answer="Правильный разрез S(U) = {(v1, v2) ∈ E | v1 ∈ U & v2 ∈ U̅} для непустого U ⊂ V."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое фундаментальная система циклов?",
            expected_answer="Множество циклов, каждый из которых содержит ровно одну хорду остова."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Чему равно цикломатическое число (циклический ранг) графа?",
            expected_answer="m(G) = q − p + 1 (число хорд остова)."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое фундаментальная система разрезов?",
            expected_answer="Множество разрезов Se, где e – ребро остова, включающее e и все хорды, соединяющие два берега."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Чему равен коциклический ранг (коцикломатическое число)?",
            expected_answer="m*(G) = p − 1 (число рёбер остова)."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Как связаны пространство циклов и пространство разрезов?",
            expected_answer="Они являются подпространствами пространства подмножеств рёбер, ортогональны друг другу."))

        # ----- 10.2 Эйлеровы циклы -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Какой граф называется эйлеровым?",
            expected_answer="Граф, имеющий цикл, содержащий все рёбра (эйлеров цикл)."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Сформулируйте критерий эйлеровости графа.",
            expected_answer="Связный граф эйлеров тогда и только тогда, когда все его вершины имеют чётную степень."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Какой граф называется полуэйлеровым?",
            expected_answer="Граф, имеющий эйлерову цепь (не обязательно замкнутую), содержащую все рёбра."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Сколько вершин нечётной степени может быть в полуэйлеровом графе?",
            expected_answer="Ровно две вершины нечётной степени."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Оцените долю эйлеровых графов среди всех графов с p вершинами.",
            expected_answer="Почти нет, предел отношения |E(p)|/|G(p)| стремится к 0 при p→∞."))

        # ----- 10.3 Гамильтоновы циклы -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое гамильтонов цикл?",
            expected_answer="Простой цикл, содержащий все вершины графа."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Сформулируйте достаточное условие гамильтоновости по теореме Оре.",
            expected_answer="Если для любой пары несмежных вершин u, v выполняется d(u)+d(v) ≥ p, то граф гамильтонов."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Является ли задача коммивояжёра NP-полной?",
            expected_answer="Да, задача коммивояжёра (поиск кратчайшего гамильтонова цикла) NP-полна."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что утверждает теорема Поша?",
            expected_answer="Если для всех n < (p-1)/2 число вершин степени ≤ n меньше n, и для нечётных p дополнительное условие, то граф гамильтонов."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Какова доля гамильтоновых графов среди всех графов при большом p?",
            expected_answer="Почти все графы гамильтоновы (предел отношения |H(p)|/|G(p)| = 1)."))

        # ----- 10.4 Независимые и покрывающие множества -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое вершинное покрытие?",
            expected_answer="Множество вершин, инцидентных всем рёбрам графа."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что называется числом вершинного покрытия α₀?",
            expected_answer="Наименьшее количество вершин в вершинном покрытии."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое независимое множество вершин?",
            expected_answer="Множество попарно несмежных вершин."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Как связаны числа α₀ и β₀ (вершинное покрытие и независимость)?",
            expected_answer="α₀ + β₀ = p."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое паросочетание?",
            expected_answer="Набор попарно несмежных рёбер."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Сформулируйте теорему Кёнига.",
            expected_answer="В двудольном графе размер максимального паросочетания равен размеру минимального вершинного покрытия (β₁ = α₀)."))

        # ----- 10.5 Построение независимых множеств -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Почему для задачи о наибольшем независимом множестве нельзя применить жадный алгоритм?",
            expected_answer="Семейство независимых множеств не образует матроид (аксиома M3 не выполняется)."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое поиск с возвратами (бэктрекинг)?",
            expected_answer="Метод перебора, при котором из текущей ситуации пробуются все возможные продолжения, а при неудаче выполняется возврат к предыдущему состоянию."))

        # ----- 10.6 Доминирующие множества -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое доминирующее множество вершин?",
            expected_answer="Множество S такое, что каждая вершина либо в S, либо смежна с S (S ∪ Γ(S) = V)."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что называется ядром графа?",
            expected_answer="Независимое доминирующее множество вершин."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Любой ли граф имеет ядро?",
            expected_answer="Да, любой неориентированный граф имеет ядро."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Сформулируйте задачу о наименьшем покрытии (Set Cover).",
            expected_answer="Дано множество V и семейство его подмножеств, найти покрытие V с минимальным суммарным весом."))

        # ----- 10.7 Раскраска графов -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что называется хроматическим числом графа?",
            expected_answer="Наименьшее число цветов, необходимых для правильной раскраски вершин."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Чему равно хроматическое число полного графа Kₚ?",
            expected_answer="χ(Kₚ) = p."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Чему равно хроматическое число двудольного графа?",
            expected_answer="2 (если есть хотя бы одно ребро)."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Как оценить хроматическое число через максимальную степень Δ?",
            expected_answer="χ(G) ≤ Δ(G) + 1."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое хроматический полином?",
            expected_answer="Полином P_G(x), значение которого при x = n равно числу правильных раскрасок в n цветов."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Чему равен хроматический полином полного графа Kₚ?",
            expected_answer="P_{Kₚ}(x) = x(x−1)...(x−p+1)."))

        # ----- 10.8 Планарность -----
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Какой граф называется планарным?",
            expected_answer="Граф, который можно нарисовать на плоскости без пересечения рёбер."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Сформулируйте формулу Эйлера для связного планарного графа.",
            expected_answer="p − q + f = 2, где f – число граней."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Почему граф K₅ не планарен?",
            expected_answer="Для K₅: p=5, q=10, но q ≤ 3p−6 = 9, противоречие."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Почему граф K₃,₃ не планарен?",
            expected_answer="В нём нет треугольников, поэтому 4f ≤ 2q, f=5 → 10 ≤ 9 – противоречие."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Сформулируйте теорему Куратовского.",
            expected_answer="Граф планарен тогда и только тогда, когда он не содержит подграфов, гомеоморфных K₅ или K₃,₃."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Теорема о пяти красках: что она утверждает?",
            expected_answer="Всякий планарный граф можно правильно раскрасить пятью цветами."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Почему в любом планарном графе есть вершина степени не больше 5?",
            expected_answer="Из следствия формулы Эйлера: если бы все степени ≥6, то 6p≤2q, откуда q≥3p, но q≤3p−6 – противоречие."))

        # Дополнительно до 50
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое фундаментальный цикл, соответствующий хорде?",
            expected_answer="Единственный цикл, содержащий данную хорду и рёбра остова."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Чему равно хроматическое число чётного цикла?",
            expected_answer="2."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Чему равно хроматическое число нечётного цикла?",
            expected_answer="3."))
        pool.append(Question(chunk_ids=[], combined_text="",
            question_text="Что такое гомеоморфизм графов?",
            expected_answer="Графы гомеоморфны, если один можно получить из другого подразбиением рёбер."))

        return pool[:50]

    def _precompute_expected_embeddings(self):
        """Вычисляет эмбеддинг для каждого эталонного ответа."""
        for q in self._static_question_pool:
            if q.expected_embedding is None:
                emb = self.embedding_model.encode(q.expected_answer, convert_to_numpy=True)
                q.expected_embedding = emb.astype(np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def evaluate_answer(self, question: Question, student_answer: str) -> Tuple[float, str]:
        """
        Оценивает ответ через косинусное сходство эмбеддингов.
        Возвращает (оценка 0 или 1, фидбек).
        """
        if not student_answer.strip():
            return 0.0, "Ответ пустой."

        # Эмбеддинг ответа студента
        student_emb = self.embedding_model.encode(student_answer, convert_to_numpy=True).astype(np.float32)
        expected_emb = question.expected_embedding

        sim = self._cosine_similarity(student_emb, expected_emb)
        if sim >= self.similarity_threshold:
            score = 1.0
            feedback = f"Правильно! Сходство с эталоном: {sim:.2%}"
        else:
            score = 0.0
            feedback = f"Неверно. Сходство с эталоном: {sim:.2%} (нужно ≥ {self.similarity_threshold:.0%})"

        return score, feedback

    def get_questions_from_pool(self, n: int) -> List[Question]:
        if n > len(self._static_question_pool):
            n = len(self._static_question_pool)
        return random.sample(self._static_question_pool, n)

    def generate_questions_for_session(self, num_questions: Optional[int] = None) -> List[Question]:
        if num_questions is None:
            num_questions = self.questions_per_session
        return self.get_questions_from_pool(num_questions)

    def run_test_session(self, num_questions: Optional[int] = None) -> Dict:
        """Консольная версия тестирования (для обратной совместимости)."""
        if num_questions is None:
            num_questions = self.questions_per_session
        questions = self.get_questions_from_pool(num_questions)
        if not questions:
            print("Нет вопросов.")
            return {}
        total_score = 0.0
        max_score = float(len(questions))
        details = []
        for i, q in enumerate(questions, 1):
            print(f"\n--- Вопрос {i} ---")
            print(q.question_text)
            student_ans = input("> ").strip()
            if student_ans.lower() == "skip":
                score = 0.0
                feedback = "Пропущен."
            else:
                score, feedback = self.evaluate_answer(q, student_ans)
                total_score += score
            print(f"Оценка: {score:.2f} / 1.00")
            print(feedback)
            print(f"Эталон: {q.expected_answer}")
            details.append({
                "question": q.question_text,
                "student_answer": student_ans,
                "expected_answer": q.expected_answer,
                "score": score,
                "feedback": feedback
            })
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO test_sessions (total_questions, total_score, max_possible_score, details) VALUES (?, ?, ?, ?)",
                (len(questions), total_score, max_score, json.dumps(details, ensure_ascii=False))
            )
        print(f"\nИтог: {total_score:.2f} / {max_score:.2f}")
        return {"total_score": total_score, "max_score": max_score}

    def show_statistics(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT session_date, total_questions, total_score, max_possible_score FROM test_sessions ORDER BY id"
            )
            rows = cur.fetchall()
        if not rows:
            print("Нет сессий.")
            return
        print("\nИстория:")
        for row in rows:
            date, qty, score, max_score = row
            percent = score / max_score * 100 if max_score > 0 else 0
            print(f"{date[:19]} | {qty} | {score:.2f}/{max_score:.2f} | {percent:.1f}%")
