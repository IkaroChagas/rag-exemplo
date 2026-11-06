// Repassa a pergunta para a API Python (api.py). Assim o navegador só fala
// com o Next.js: sem CORS e sem expor o endereço da API.
const RAG_API_URL = process.env.RAG_API_URL ?? "http://127.0.0.1:8000";

export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  if (!body || typeof body.question !== "string" || !body.question.trim()) {
    return Response.json({ detail: "Pergunta vazia." }, { status: 400 });
  }

  try {
    const res = await fetch(`${RAG_API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: body.question,
        so_busca: Boolean(body.so_busca),
      }),
      cache: "no-store",
    });
    const data = await res.json().catch(() => ({ detail: "Resposta inválida da API." }));
    return Response.json(data, { status: res.status });
  } catch {
    return Response.json(
      {
        detail: `Não consegui falar com a API Python em ${RAG_API_URL}. Ela está rodando? (uvicorn api:app --port 8000)`,
      },
      { status: 502 },
    );
  }
}
