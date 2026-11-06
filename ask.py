"""
Fase 2 do RAG: consulta.
Gera o embedding da pergunta, busca os chunks mais parecidos e pede a um LLM
(Claude ou Gemini) uma resposta: sobre a agência, só com base nos trechos;
perguntas gerais de turismo podem usar conhecimento geral, com aviso.

Uso:
  python ask.py "posso remarcar se chover?"
  python ask.py --so-busca "posso remarcar se chover?"   # mostra só a recuperação

Provedor do LLM: variável LLM_PROVIDER ("claude" ou "gemini").
Se não for definida, usa Gemini quando GEMINI_API_KEY existir, senão Claude.
"""
import os
import sys

from common import embed_query, get_conn

TOP_K = 6  # trechos entregues ao LLM; mais trechos = perguntas que cruzam assuntos
# "vector" (padrão): só significado. "hybrid": significado + palavras-chave
# (busca textual do Postgres). No avaliar_busca.py a híbrida foi pior com o
# modelo E5, mas pode ajudar com documentos cheios de códigos e nomes próprios.
SEARCH_MODE = os.getenv("SEARCH_MODE", "vector")
CANDIDATES = 20  # quantos candidatos cada busca traz antes de combinar
RRF_K = 60  # constante padrão do Reciprocal Rank Fusion
MAX_TOKENS = 800
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Cidade/região da agência (opcional). Ajuda nas perguntas gerais de turismo;
# sem ela o assistente não presume onde a agência fica.
DESTINO = os.getenv("DESTINO", "").strip()

SYSTEM_PROMPT = f"""Você é o assistente de atendimento de uma agência de passeios{f" em {DESTINO}" if DESTINO else ""}.
Responda em português, de forma direta e cordial.

Há dois tipos de pergunta, com regras diferentes:

1. Sobre a agência: passeios, preços, horários, reservas, pagamento, cancelamento,
   regras, segurança, contato e qualquer serviço da agência.
   - Use SOMENTE os trechos em <contexto>. Nunca invente nem estime valores,
     horários, prazos, regras ou passeios que não estejam neles.
   - Se a informação não estiver nos trechos, diga que não a encontrou e
     sugira falar com a central pelo WhatsApp.
   - Cite entre colchetes os arquivos usados, ex.: [cancelamento.md].

2. Gerais de turismo, não cobertas pelos trechos: praias, gastronomia típica,
   clima, cultura, o que fazer, dicas de viagem.
   - Pode responder com conhecimento geral, mas avise claramente, numa frase
     curta, que é uma informação geral e não da agência.
   - {f"Considere que a agência fica em {DESTINO}." if DESTINO else "Os documentos não dizem em que cidade a agência fica: não presuma. Se a resposta depender do lugar, responda de forma geral ou pergunte ao cliente onde ele está."}
   - Não recomende outras agências nem passeios que a agência não vende.

Se a pergunta misturar os dois tipos, separe as partes.

Estilo: vá direto à resposta. Não comece com saudação ("Olá!", "Tudo bem?")
nem termine oferecendo mais ajuda, a menos que a mensagem do cliente seja só
um cumprimento ou agradecimento; nesse caso, responda em uma frase curta.
Cite apenas os arquivos de onde tirou alguma informação."""


def retrieve(question: str, k: int = TOP_K, mode: str | None = None):
    """Devolve os k chunks mais relevantes: [(source, content, similaridade)]."""
    mode = mode or SEARCH_MODE
    q_vec = embed_query(question)
    conn = get_conn()

    if mode == "vector":
        # Só significado: os k chunks com menor distância de cosseno.
        return conn.execute(
            """
            SELECT source, content, 1 - (embedding <=> %(v)s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %(v)s
            LIMIT %(k)s
            """,
            {"v": q_vec, "k": k},
        ).fetchall()

    # Híbrida: duas listas ordenadas (por significado e por palavras-chave),
    # combinadas por Reciprocal Rank Fusion: cada chunk ganha 1/(60 + posição)
    # em cada lista onde aparece. Quem vai bem nas duas sobe.
    #
    # A busca por palavras usa OU entre as palavras da pergunta (já sem
    # stopwords e reduzidas ao radical pelo dicionário "portuguese"), para
    # não exigir que todas apareçam no mesmo trecho.
    return conn.execute(
        """
        WITH
        kw_query AS (
            SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS q
            FROM unnest(to_tsvector('portuguese', %(q)s))
        ),
        by_vector AS (
            SELECT id, row_number() OVER (ORDER BY embedding <=> %(v)s) AS pos
            FROM chunks
            ORDER BY embedding <=> %(v)s
            LIMIT %(n)s
        ),
        by_keyword AS (
            SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.tsv, kq.q) DESC) AS pos
            FROM chunks c, kw_query kq
            WHERE c.tsv @@ kq.q
            ORDER BY ts_rank_cd(c.tsv, kq.q) DESC
            LIMIT %(n)s
        )
        SELECT c.source, c.content, 1 - (c.embedding <=> %(v)s) AS similarity
        FROM chunks c
        LEFT JOIN by_vector bv ON bv.id = c.id
        LEFT JOIN by_keyword bk ON bk.id = c.id
        WHERE bv.id IS NOT NULL OR bk.id IS NOT NULL
        ORDER BY COALESCE(1.0 / (%(rrf)s + bv.pos), 0)
               + COALESCE(1.0 / (%(rrf)s + bk.pos), 0) DESC
        LIMIT %(k)s
        """,
        {"q": question, "v": q_vec, "n": CANDIDATES, "k": k, "rrf": RRF_K},
    ).fetchall()


def build_prompt(question: str, hits) -> str:
    contexto = "\n\n".join(
        f'<trecho fonte="{src}">\n{content}\n</trecho>' for src, content, _ in hits
    )
    return f"<contexto>\n{contexto}\n</contexto>\n\nPergunta do cliente: {question}"


def pick_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "claude"
    if provider not in ("claude", "gemini"):
        raise ValueError(f'LLM_PROVIDER inválido: "{provider}". Use "claude" ou "gemini".')
    return provider


def generate(prompt: str) -> tuple[str, str]:
    """Chama o provedor configurado. Devolve (modelo, resposta)."""
    if pick_provider() == "gemini":
        return GEMINI_MODEL, generate_gemini(prompt)
    return CLAUDE_MODEL, generate_claude(prompt)


def generate_claude(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def generate_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    # Lê GEMINI_API_KEY do ambiente. Limita as novas tentativas: quando o modelo
    # está sobrecarregado (503) o padrão do SDK fica minutos tentando de novo.
    client = genai.Client(
        http_options=types.HttpOptions(
            timeout=60_000,  # ms
            retry_options=types.HttpRetryOptions(attempts=2),
        )
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            # Nos modelos Gemini com "thinking", o raciocínio interno consome
            # parte desse limite; com 800 a resposta pode sair cortada ou vazia.
            max_output_tokens=MAX_TOKENS * 4,
        ),
    )
    return response.text or "(o Gemini não devolveu texto)"


def main():
    args = sys.argv[1:]
    so_busca = "--so-busca" in args
    args = [a for a in args if a != "--so-busca"]
    if not args:
        print(__doc__)
        sys.exit(1)
    question = " ".join(args)

    hits = retrieve(question)

    print("\n--- Trechos recuperados ---")
    for src, content, sim in hits:
        preview = content.replace("\n", " ")[:90]
        print(f"[{sim:.2f}] {src}: {preview}...")

    if so_busca:
        return

    try:
        model, answer = generate(build_prompt(question, hits))
    except ValueError as e:
        sys.exit(str(e))

    print(f"\n--- Resposta ({model}) ---")
    print(answer)


if __name__ == "__main__":
    main()
