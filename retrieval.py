import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Tuple
from parser import extract_text_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS не установлен. Установите 'pip install faiss-cpu' для ускорения поиска.")


class PDFVectorIndexer:
    def __init__(
            self,
            pdf_path: str,
            db_path: str = "vectors.db",
            model_name: str = "T-lite-it-2.1-Q4_K_M.gguf",
            batch_size: int = 1,
            resume: bool = True,
            min_text_length: int = 100,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            use_faiss: bool = True,
            faiss_index_path: str = "faiss.index"
    ):
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.resume = resume
        self.min_text_length = min_text_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index_path = faiss_index_path

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        self._init_db()

        # FAISS: индекс и маппинг (загружаются при наличии)
        self._faiss_index = None
        self._id_to_chunk = []   # список (page_number, chunk_index, chunk_text)
        if self.use_faiss:
            self._load_or_build_faiss()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    chunk_text TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page ON chunk_embeddings(page_number)")

    def _is_page_processed(self, page_num: int) -> bool:
        if not self.resume:
            return False
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT 1 FROM chunk_embeddings WHERE page_number = ? LIMIT 1",
                (page_num,)
            )
            return cur.fetchone() is not None

    def _embed_text(self, text: str) -> bytes:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32).tobytes()

    def _save_chunk(self, page_num: int, chunk_index: int, chunk_text: str, embedding_blob: bytes):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chunk_embeddings (page_number, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                (page_num, chunk_index, chunk_text, embedding_blob)
            )

    def process(self):
        page_generator = extract_text_pages(self.pdf_path)
        page_num = 1
        while True:
            try:
                text = next(page_generator)
            except StopIteration:
                break
            if self._is_page_processed(page_num):
                print(f"Страница {page_num} уже обработана, пропускаем.")
                page_num += 1
                continue
            stripped_text = text.strip()
            if len(stripped_text) < self.min_text_length:
                print(f"Страница {page_num} слишком короткая, пропускаем.")
                page_num += 1
                continue
            print(f"Обработка страницы {page_num}...")
            chunks = self.text_splitter.split_text(stripped_text)
            for idx, chunk in enumerate(chunks):
                embedding_blob = self._embed_text(chunk)
                self._save_chunk(page_num, idx, chunk, embedding_blob)
            print(f"Страница {page_num} сохранена (чанков: {len(chunks)}).")
            page_num += 1
        print("Обработка завершена.")

    # -------------------- FAISS методы --------------------
    def _load_or_build_faiss(self):
        """Загружает готовый FAISS индекс или строит его из БД."""
        if Path(self.faiss_index_path).exists():
            try:
                self._faiss_index = faiss.read_index(self.faiss_index_path)
                self._load_mapping()
                print(f"FAISS индекс загружен, размер: {self._faiss_index.ntotal}")
                return
            except Exception as e:
                print(f"Не удалось загрузить FAISS индекс: {e}, перестроим.")
        self._build_faiss_index()

    def _build_faiss_index(self):
        """Читает все эмбеддинги из БД и строит плоский индекс."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT page_number, chunk_index, chunk_text, embedding FROM chunk_embeddings")
            all_embeddings = []
            self._id_to_chunk.clear()
            for page, idx, text, blob in cur:
                emb = np.frombuffer(blob, dtype=np.float32)
                all_embeddings.append(emb)
                self._id_to_chunk.append((page, idx, text))
        if not all_embeddings:
            print("Нет эмбеддингов для построения FAISS индекса.")
            self.use_faiss = False
            return
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        dimension = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dimension)   # inner product (косинус после нормировки)
        faiss.normalize_L2(embeddings)
        self._faiss_index.add(embeddings)
        faiss.write_index(self._faiss_index, self.faiss_index_path)
        self._save_mapping()
        print(f"Построен FAISS индекс, чанков: {self._faiss_index.ntotal}")

    def _save_mapping(self):
        with open(self.faiss_index_path + ".meta", "w", encoding="utf-8") as f:
            for page, idx, text in self._id_to_chunk:
                # избавляемся от символов-разделителей
                text_clean = text.replace('\n', ' ').replace('|', ' ')
                f.write(f"{page}|{idx}|{text_clean}\n")

    def _load_mapping(self):
        self._id_to_chunk.clear()
        meta_path = self.faiss_index_path + ".meta"
        if not Path(meta_path).exists():
            return
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 2)
                if len(parts) == 3:
                    page, idx, text = parts
                    self._id_to_chunk.append((int(page), int(idx), text))

    # -------------------- Поиск (с поддержкой FAISS) --------------------
    def search_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[int, int, str, float]]:
        """Быстрый поиск через FAISS (или медленный fallback)."""
        if self.use_faiss and self._faiss_index is not None:
            query_emb = self.model.encode(query_text, convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(query_emb.reshape(1, -1))
            scores, indices = self._faiss_index.search(query_emb.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    break
                page, chunk_idx, text = self._id_to_chunk[idx]
                results.append((page, chunk_idx, text, float(score)))
            return results
        else:
            # fallback: медленный линейный поиск
            return self._legacy_search_similar(query_text, top_k)

    def _legacy_search_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[int, int, str, float]]:
        """Старый медленный поиск, используется только если FAISS недоступен."""
        query_emb = self.model.encode(query_text, convert_to_numpy=True).astype(np.float32)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT page_number, chunk_index, chunk_text, embedding FROM chunk_embeddings")
            results = []
            for page_num, chunk_idx, chunk_text, blob in cur.fetchall():
                emb = np.frombuffer(blob, dtype=np.float32)
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                results.append((page_num, chunk_idx, chunk_text, sim))
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    # Вспомогательные методы для совместимости
    def get_chunks_by_page(self, page_num: int) -> List[Tuple[int, str, np.ndarray]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT chunk_index, chunk_text, embedding FROM chunk_embeddings WHERE page_number = ? ORDER BY chunk_index",
                (page_num,)
            )
            results = []
            for idx, text, blob in cur.fetchall():
                emb = np.frombuffer(blob, dtype=np.float32)
                results.append((idx, text, emb))
            return results

    def get_page_text(self, page_num: int) -> Optional[str]:
        chunks = self.get_chunks_by_page(page_num)
        if not chunks:
            return None
        return "\n".join([text for _, text, _ in chunks])