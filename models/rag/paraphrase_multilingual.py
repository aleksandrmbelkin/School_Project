from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import re, os, json
from pathlib import Path
from typing import List, Dict


def read_txt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    text = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text)

def split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in paras]
    paras = [p for p in paras if len(p) > 20]
    return paras

def chunk_long(p: str, max_len: int = 500, overlap: int = 50) -> List[str]:
    if len(p) <= max_len:
        return [p]
    chunks = []
    start = 0
    while start < len(p):
        end = min(start + max_len, len(p))
        chunks.append(p[start:end])
        if end == len(p):
            break
        start = end - overlap
    return chunks

def load_corpus_from_uploads(manual_paths: List[str]) -> List[Dict]:
    docs = []

    for p in manual_paths:
        path = Path(p)
        if not path.exists():
            print(f"Файл не найден: {p}")
            continue
        ext = path.suffix.lower()
        if ext == ".txt":
            text = read_txt(path)
        elif ext == ".pdf":
            text = read_pdf(path)
        else:
            print(f"Пропущен {p}: поддерживаются только .txt и .pdf")
            continue
        for i, para in enumerate(split_paragraphs(text)):
            for j, chunk in enumerate(chunk_long(para)):
                docs.append({"source": str(path.name), "para_id": i, "chunk_id": j, "text": chunk})

    return docs

def embed_texts(texts: List[str]) -> np.ndarray:
    return model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def search(query: str, top_k: int = 5):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = emb @ q  # косинусные близости для всех документов
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        d = docs[i]
        results.append({
            "score": float(sims[i]),
            "source": d["source"],
            "para_id": d["para_id"],
            "chunk_id": d["chunk_id"],
            "text": d["text"],
        })
    return results


def load_index():
    global emb, docs
    data = np.load("embeddings_and_docs.npz")
    emb = data["emb"]
    with open("docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    print("Индекс загружен:", emb.shape, "фрагментов")


if not os.path.isdir('./models/rag/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'):
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name, cache_folder='./models/rag')

manual_paths = []

docs = load_corpus_from_uploads(manual_paths)
print(f"Загружено фрагментов: {len(docs)}")
print("Пример записи:", docs[0] if docs else None)

texts = [d["text"] for d in docs]
emb = embed_texts(texts)
print("Готово! Размер эмбеддингов:", emb.shape)

user_query = "Напишите свой вопрос сюда"
for r in search(user_query, top_k=5):
    print(f"- score={r['score']:.3f} | source={r['source']} (para {r['para_id']}, chunk {r['chunk_id']})\n  {r['text'][:500]}...\n")


np.savez("embeddings_and_docs.npz", emb=emb)
with open("docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)
print("Сохранено: embeddings_and_docs.npz и docs.json")
