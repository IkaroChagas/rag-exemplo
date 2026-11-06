"""
Fase 1 do RAG: indexação.
Lê os arquivos de docs/, quebra em chunks, gera embeddings e salva no pgvector.

Uso: python ingest.py
"""
from pathlib import Path

from common import embed_passages, embedding_dim, get_conn

DOCS_DIR = Path(__file__).parent / "docs"
MAX_CHARS = 1200  # teto por chunk (~300 tokens), abaixo do limite do modelo


def parse_markdown(text: str):
    """Separa o documento em (título, [(seção, corpo), ...]).

    O título é a linha "# ...". Cada "## ..." abre uma seção; o que vem antes
    da primeira seção fica com seção None.
    """
    title, sections = None, []
    heading, body = None, []

    for line in text.splitlines():
        if line.startswith("# ") and title is None:
            title = line[2:].strip()
        elif line.startswith("## "):
            if "".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    return title, sections


def chunk_document(text: str) -> list[str]:
    """Um chunk por seção (ou por parágrafo, quando não há seções).

    Cada chunk começa com "Título do documento — Seção". Esse cabeçalho dá
    contexto ao embedding: um trecho que diz só "Adulto: R$ 180" passa a
    dizer de qual passeio e de qual documento é.
    """
    title, sections = parse_markdown(text)
    chunks = []
    for heading, body in sections:
        context = " — ".join(p for p in (title, heading) if p)
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

        if heading is None:
            # Sem seção (ex.: FAQ): cada parágrafo é um assunto independente.
            groups = [[p] for p in paragraphs]
        else:
            # Com seção: mantém a seção inteira, quebrando só se for longa.
            groups, current = [], []
            for p in paragraphs:
                if current and len("\n\n".join(current + [p])) > MAX_CHARS:
                    groups.append(current)
                    current = []
                current.append(p)
            if current:
                groups.append(current)

        for g in groups:
            chunks.append(f"{context}\n\n" + "\n\n".join(g))
    return chunks


def main():
    conn = get_conn()
    conn.execute("DROP TABLE IF EXISTS chunks")
    # tsv: versão "por palavras" do texto, usada na busca por palavra-chave
    # (a busca híbrida combina ela com a busca por significado).
    conn.execute(f"""
        CREATE TABLE chunks (
            id        bigserial PRIMARY KEY,
            source    text NOT NULL,
            content   text NOT NULL,
            embedding vector({embedding_dim()}) NOT NULL,
            tsv       tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED
        )
    """)

    rows = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for chunk in chunk_document(path.read_text(encoding="utf-8")):
            rows.append((path.name, chunk))

    print(f"Gerando embeddings de {len(rows)} chunks...")
    vectors = embed_passages([content for _, content in rows])

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s)",
            [(src, content, vec) for (src, content), vec in zip(rows, vectors)],
        )

    # Índice HNSW para busca aproximada rápida e GIN para a busca por palavras.
    # Com poucos chunks não faz diferença, mas é o que você usaria com
    # milhares/milhões de linhas.
    conn.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)")
    conn.execute("CREATE INDEX ON chunks USING gin (tsv)")
    print("Pronto. Rode: python ask.py \"sua pergunta\"")


if __name__ == "__main__":
    main()
