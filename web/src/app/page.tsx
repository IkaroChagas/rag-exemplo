"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type Hit = { source: string; content: string; similarity: number };

type Turn = {
  id: number;
  question: string;
  soBusca: boolean;
  status: "loading" | "done" | "error";
  hits?: Hit[];
  answer?: string | null;
  model?: string | null;
  error?: string;
};

const EXEMPLOS = [
  "Se chover, meu passeio de barco é cancelado?",
  "Quanto custa o catamarã para 2 adultos e uma criança de 8 anos?",
  "Grávida pode fazer mergulho?",
  "O guia fala inglês?",
  "Esqueci minha câmera na van, e agora?",
  "Quais comidas típicas do Nordeste eu deveria provar?",
  "Qual o preço do passeio de buggy?",
];

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [soBusca, setSoBusca] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  const busy = turns.some((t) => t.status === "loading");

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const id = nextId.current++;
    setInput("");
    setTurns((ts) => [...ts, { id, question: q, soBusca, status: "loading" }]);

    const update = (patch: Partial<Turn>) =>
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)));

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, so_busca: soBusca }),
      });
      const data = await res.json();
      if (!res.ok) {
        update({ status: "error", error: formatError(data?.detail) });
        return;
      }
      update({ status: "done", hits: data.hits, answer: data.answer, model: data.model });
    } catch {
      update({ status: "error", error: "Falha de rede ao falar com o servidor." });
    }
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3">
          <div>
            <h1 className="text-base font-semibold">Assistente de passeios</h1>
            <p className="text-xs text-muted">
              Sobre a agência, responde pelos documentos (RAG); dúvidas gerais de turismo vêm com aviso
            </p>
          </div>
          {turns.length > 0 && (
            <button
              onClick={() => setTurns([])}
              disabled={busy}
              className="rounded-md px-2.5 py-1.5 text-xs text-muted hover:bg-accent-soft hover:text-foreground disabled:opacity-50"
            >
              Limpar conversa
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-6">
          {turns.length === 0 ? (
            <EmptyState onPick={ask} />
          ) : (
            turns.map((t) => <TurnView key={t.id} turn={t} />)
          )}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="border-t border-border bg-surface">
        <form
          className="mx-auto flex max-w-3xl flex-col gap-2 px-4 py-3"
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
        >
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              rows={1}
              placeholder="Pergunte sobre passeios, cancelamento, pagamento…"
              className="max-h-40 min-h-11 flex-1 resize-none rounded-lg border border-border bg-background px-3 py-2.5 text-sm outline-none focus:border-accent"
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              className="h-11 rounded-lg bg-accent px-4 text-sm font-medium text-white disabled:opacity-40 dark:text-black"
            >
              Enviar
            </button>
          </div>
          <label className="flex w-fit cursor-pointer items-center gap-2 text-xs text-muted">
            <input
              type="checkbox"
              checked={soBusca}
              onChange={(e) => setSoBusca(e.target.checked)}
              className="accent-(--accent)"
            />
            Só busca: mostra os trechos encontrados sem chamar o LLM
          </label>
        </form>
      </footer>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center gap-5 pt-10 text-center">
      <div>
        <h2 className="text-lg font-semibold">Como posso ajudar?</h2>
        <p className="mt-1 max-w-md text-sm text-muted">
          A pergunta vira um vetor e os trechos mais parecidos dos documentos são buscados
          no pgvector. Sobre a agência, o LLM responde só com esses trechos; dúvidas gerais de
          turismo ele responde com conhecimento geral, avisando.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {EXEMPLOS.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="rounded-full border border-border bg-surface px-3 py-1.5 text-sm hover:border-accent hover:text-accent"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="max-w-[85%] self-end whitespace-pre-wrap rounded-2xl rounded-br-sm bg-user-bubble px-4 py-2.5 text-sm text-user-text">
        {turn.question}
      </div>

      <div className="flex flex-col gap-2 rounded-2xl rounded-bl-sm border border-border bg-surface px-4 py-3">
        {turn.status === "loading" && (
          <p className="animate-pulse text-sm text-muted">
            {turn.soBusca ? "Buscando nos documentos…" : "Buscando nos documentos e gerando a resposta…"}
          </p>
        )}

        {turn.status === "error" && (
          <p className="rounded-md bg-danger-soft px-3 py-2 text-sm text-danger">{turn.error}</p>
        )}

        {turn.status === "done" && (
          <>
            {turn.answer ? (
              <Answer text={turn.answer} />
            ) : (
              <p className="text-sm text-muted">Modo só busca: o LLM não foi chamado.</p>
            )}
            {turn.model && <p className="text-[11px] text-muted">Respondido por {turn.model}</p>}
            {turn.hits && <Sources hits={turn.hits} open={!turn.answer} />}
          </>
        )}
      </div>
    </div>
  );
}

// Citações do modelo: [precos.md] ou [dicas.md, passeios.md].
const CITATION = /\[\s*([\w.\-]+\.md(?:\s*,\s*[\w.\-]+\.md)*)\s*\]/g;

// O LLM responde em markdown (negrito, listas). As citações viram código
// inline, que aqui é estilizado como etiqueta.
function Answer({ text }: { text: string }) {
  const md = text.replace(CITATION, (_, files: string) =>
    files
      .split(",")
      .map((f) => `\`${f.trim()}\``)
      .join(" "),
  );
  return (
    <div className="text-sm leading-relaxed [&_li]:my-0.5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_strong]:font-semibold [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5">
      <ReactMarkdown
        components={{
          code: ({ children }) => (
            <span className="mx-0.5 rounded bg-accent-soft px-1.5 py-0.5 font-mono text-[11px] text-accent">
              {children}
            </span>
          ),
        }}
      >
        {md}
      </ReactMarkdown>
    </div>
  );
}

function Sources({ hits, open }: { hits: Hit[]; open: boolean }) {
  return (
    <details open={open} className="group mt-1 border-t border-border pt-2">
      <summary className="cursor-pointer select-none text-xs text-muted hover:text-foreground">
        Trechos recuperados ({hits.length})
      </summary>
      <ol className="mt-2 flex flex-col gap-2">
        {hits.map((h, i) => (
          <li key={i} className="rounded-lg border border-border bg-background p-3">
            <div className="mb-1.5 flex items-center gap-2">
              <span className="font-mono text-xs font-medium">{h.source}</span>
              <div className="ml-auto flex items-center gap-1.5" title="Similaridade de cosseno com a pergunta">
                <div className="h-1.5 w-16 overflow-hidden rounded-full bg-border">
                  <div
                    className="h-full rounded-full bg-accent"
                    style={{ width: `${Math.max(0, Math.min(1, h.similarity)) * 100}%` }}
                  />
                </div>
                <span className="w-8 text-right font-mono text-[11px] text-muted">
                  {h.similarity.toFixed(2)}
                </span>
              </div>
            </div>
            <p className="whitespace-pre-wrap text-xs leading-relaxed text-muted">{h.content}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}

// A API já devolve mensagens legíveis em "detail".
function formatError(detail: unknown): string {
  return typeof detail === "string" ? detail : "Erro inesperado.";
}
