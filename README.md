# RAG de exemplo: assistente de agência de passeios

Um assistente que responde perguntas de clientes de uma agência de passeios usando RAG (Retrieval-Augmented Generation):

1. **Indexação** (`ingest.py`): os documentos em `docs/` são quebrados por seção, viram vetores com um modelo de embeddings local e são gravados no Postgres com pgvector.
2. **Consulta** (`ask.py`): a pergunta vira um vetor, os trechos mais parecidos são buscados no banco e um LLM (Gemini ou Claude) escreve a resposta com base neles.

Inclui interface de chat em Next.js e um script que mede a qualidade da busca.

Os documentos em `docs/` são **fictícios**: simulam a base de conhecimento de uma agência (passeios, preços, reservas, cancelamento, segurança, dicas, atendimento).

## Pré-requisitos

- Python 3.11+
- Docker (para o Postgres com pgvector)
- Node.js 20+ (só para a interface web)
- Uma chave de API do [Gemini](https://aistudio.google.com/apikey) (tem cota gratuita) ou do [Claude](https://console.anthropic.com)

## Rodando

```bash
# 1. Banco com pgvector
docker compose up -d

# 2. Dependências
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configuração: copie o exemplo e coloque sua chave em GEMINI_API_KEY ou ANTHROPIC_API_KEY
cp .env.example .env

# 4. Indexar os documentos (baixa o modelo de embeddings na 1ª vez, ~470 MB)
#    Rode de novo sempre que mudar algo em docs/
python ingest.py

# 5. Perguntar pelo terminal
python ask.py "se chover meu passeio de barco é cancelado?"
python ask.py "qual o preço do passeio de buggy?"   # não está nos docs: deve dizer que não encontrou
python ask.py --so-busca "posso levar meu cachorro?" # só mostra os trechos achados, sem chamar o LLM
```

Se a porta 5432 já estiver ocupada por outro Postgres, troque a porta em `docker-compose.yml` (ex.: `"5433:5432"`) e ajuste `DATABASE_URL` no `.env`.

## Interface web (Next.js)

Dois processos, cada um em um terminal:

```bash
# API Python (busca + LLM), usa o mesmo .env
uvicorn api:app --port 8000

# Interface em http://localhost:3000
cd web
npm install   # só na 1ª vez
npm run dev
```

O Next.js repassa as perguntas para a API em `RAG_API_URL` (padrão `http://127.0.0.1:8000`). A tela mostra a resposta, os arquivos citados e os trechos recuperados com a similaridade de cada um.

## Como ele responde

- **Sobre a agência** (passeios, preços, regras, reservas): só com base nos trechos encontrados em `docs/`. Se não estiver lá, diz que não encontrou e indica a central. Não inventa valores nem regras.
- **Gerais de turismo** (comida típica, praias, clima): pode usar conhecimento geral, avisando que não é informação da agência. Defina `DESTINO` no `.env` (ex.: `DESTINO=João Pessoa, PB`) para ele saber onde a agência fica.

## Configuração (`.env`)

| Variável | Padrão | Para que serve |
|---|---|---|
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | | Chave do LLM |
| `LLM_PROVIDER` | `gemini` se houver chave do Gemini, senão `claude` | Qual LLM usar |
| `GEMINI_MODEL` / `CLAUDE_MODEL` | `gemini-3.5-flash-lite` / `claude-sonnet-5-5` | Modelo de cada provedor |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Conexão com o Postgres |
| `DESTINO` | vazio | Cidade da agência, para perguntas gerais |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modelo de embeddings (ao trocar, rode `ingest.py` de novo) |
| `SEARCH_MODE` | `vector` | `vector` (significado) ou `hybrid` (significado + palavras-chave) |

## Qualidade da busca

`python avaliar_busca.py` roda 36 perguntas de teste, escritas com palavras diferentes das usadas nos documentos, e mostra quantas encontraram o trecho certo. Não chama o LLM, então não gasta cota. Use para comparar mudanças de chunking, modelo de embeddings ou tipo de busca.

Resultados com os documentos atuais:

| Configuração | Trecho certo entre os 6 | Trecho certo em 1º |
|---|---|---|
| Chunks de ~600 caracteres + `paraphrase-multilingual-MiniLM-L12-v2` | 89% | — |
| Chunks por seção + MiniLM | 97% | 61% |
| **Chunks por seção + `multilingual-e5-small` (padrão)** | 94% | **89%** |
| Chunks por seção + e5 + busca híbrida | 94% | 75% |

O maior ganho veio do chunking: o MiniLM só lê 128 tokens, e a maioria dos chunks antigos era cortada no embedding. O e5 lê até 512 tokens e coloca o trecho certo em primeiro lugar com muito mais frequência.

## Arquivos

- `docs/`: base de conhecimento (fictícia). Para usar seus próprios documentos, coloque arquivos `.md` aqui (de preferência com seções `## título`) e rode `python ingest.py`
- `common.py`: conexão com o banco e geração de embeddings
- `ingest.py`: chunking por seção, embeddings e gravação no pgvector
- `ask.py`: busca, montagem do prompt e chamada ao Gemini ou Claude
- `avaliar_busca.py`: mede a qualidade da busca com perguntas de teste
- `api.py`: API HTTP (FastAPI) com as mesmas funções do `ask.py`, usada pela interface
- `web/`: interface de chat em Next.js (`src/app/page.tsx`) e rota que repassa para a API (`src/app/api/ask/route.ts`)
