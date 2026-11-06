"""
API HTTP do RAG, usada pela interface web (pasta web/).

Reaproveita as mesmas funções do ask.py: busca no pgvector + geração com o LLM.

Uso:
  uvicorn api:app --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ask import build_prompt, generate, retrieve
from common import get_embedder

# Registra cada pergunta, os trechos achados e a resposta no terminal da API.
log = logging.getLogger("rag")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
log.addHandler(_handler)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Carrega o modelo de embeddings ao subir, e não na 1ª pergunta.
    get_embedder()
    yield


app = FastAPI(title="RAG exemplo", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    so_busca: bool = False  # True = só a recuperação, sem chamar o LLM


class Hit(BaseModel):
    source: str
    content: str
    similarity: float


class AskResponse(BaseModel):
    hits: list[Hit]
    answer: str | None = None
    model: str | None = None


def friendly_error(e: Exception) -> str:
    """Traduz os erros mais comuns dos provedores para uma mensagem legível."""
    msg = str(e)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg:
        return ("Limite de uso do LLM atingido (no Gemini gratuito são 15 perguntas "
                "por minuto). Espere cerca de 30 segundos e tente de novo.")
    if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg:
        return "O modelo está sobrecarregado no momento. Tente de novo em instantes."
    if "API key" in msg or "API_KEY" in msg or "401" in msg or "authentication" in msg:
        return "Chave de API ausente ou inválida. Confira o arquivo .env."
    return f"Falha ao gerar a resposta: {msg[:300]}"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Pergunta vazia.")

    try:
        rows = retrieve(question)
    except Exception as e:  # banco fora do ar, tabela não indexada etc.
        raise HTTPException(503, f"Falha na busca: {e}")

    hits = [Hit(source=s, content=c, similarity=float(sim)) for s, c, sim in rows]
    top = ", ".join(f"{h.source}:{h.similarity:.2f}" for h in hits)
    log.info("PERGUNTA %r | so_busca=%s | trechos: %s", question, req.so_busca, top)
    if req.so_busca:
        return AskResponse(hits=hits)

    try:
        model, answer = generate(build_prompt(question, rows))
    except Exception as e:  # chave ausente/inválida, modelo sobrecarregado etc.
        log.error("ERRO ao gerar resposta para %r: %s", question, e)
        raise HTTPException(502, friendly_error(e))

    log.info("RESPOSTA (%s) %r", model, answer[:300])
    return AskResponse(hits=hits, answer=answer, model=model)
