"""Configuração compartilhada: conexão com o banco e modelo de embeddings."""
import os

from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (se existir). Variáveis já definidas
# no terminal têm prioridade sobre as do arquivo.
load_dotenv()

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

# Modelo multilíngue treinado para busca, roda local e não precisa de chave.
# Lê até 512 tokens por trecho e gera vetores de 384 dimensões.
# Ao trocar de modelo, rode `python ingest.py` de novo: vetores de modelos
# diferentes não são comparáveis.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embedding_dim() -> int:
    return get_embedder().get_sentence_embedding_dimension()


# Os modelos E5 foram treinados com prefixos que dizem se o texto é uma
# pergunta ("query: ") ou um trecho a ser encontrado ("passage: ").
# Sem eles a qualidade da busca cai.
def _prefix(kind: str) -> str:
    return f"{kind}: " if "e5" in EMBEDDING_MODEL.lower() else ""


def _embed(texts: list[str], kind: str):
    # normalize_embeddings=True deixa os vetores com norma 1,
    # então distância de cosseno fica direta de interpretar.
    p = _prefix(kind)
    return get_embedder().encode([p + t for t in texts], normalize_embeddings=True)


def embed_query(text: str):
    return _embed([text], "query")[0]


def embed_passages(texts: list[str]):
    return _embed(texts, "passage")


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn
