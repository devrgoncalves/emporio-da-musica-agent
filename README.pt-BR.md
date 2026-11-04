# Empório da Música — Agente de Atendimento

> 🇺🇸 [Read in English](README.md)

Protótipo em Python de um agente com LLM que atende clientes da **Empório da Música**, uma loja
fictícia de instrumentos musicais em Campo Grande/MS. Ele fala como o assistente da loja ("Tom") e
decide sozinho quando **consultar dados** (catálogo, estoque, promoções, pedidos) e quando **ler o
manual de políticas** (horários, pagamento, trocas, frete, garantia). Também lida com perguntas fora
do escopo da loja.

**Stack:** Python 3.10+ · LangChain `create_agent` (sobre o LangGraph) · Google Gemini · pandas · Streamlit · pytest

---

## 1. Como rodar

```bash
# 1. Crie um ambiente virtual e instale as dependências
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt

# 2. Configure o provedor do modelo (chave gratuita: https://aistudio.google.com/apikey)
cp .env.example .env              # Windows: copy .env.example .env
# edite o .env e preencha GOOGLE_API_KEY=...

# 3. Suba a interface de chat  ->  http://localhost:8501
streamlit run app.py
```

| Comando | O que faz |
|---|---|
| `streamlit run app.py` | Interface de chat. O botão na barra lateral mostra quais ferramentas o agente chamou e o que retornaram. |
| `pytest -m "not llm"` | Testes unitários (dados, ferramentas, prompt). Não precisa de chave. |
| `pytest -m llm` | Testes ponta a ponta com a API real do Gemini. |
| `python scripts/run_examples.py` | Gera de novo as conversas de exemplo em [`examples/`](examples/). |

Variáveis de ambiente (`.env`): `GOOGLE_API_KEY` (obrigatória) e `GEMINI_MODEL` (opcional, padrão
`gemini-3.5-flash-lite`).

---

## 2. Arquitetura

```
 Cliente ──digita──► Streamlit (app.py) ──chat(thread_id, texto)──► Agente (create_agent + Gemini)
                                                                     │  system prompt: persona, regras,
                                                                     │  data/hora atual da loja
                                                                     │  memória: InMemorySaver por conversa
                       o modelo decide qual ferramenta chamar (function calling, laço ReAct)
          ┌──────────────────────────────┬──────────────────────────┬──────────────────────────┐
   search_products(query, category,  check_order(order_id,       get_policy(topic)
   max_price, min_price)             email_or_phone)             topic ∈ {about, hours, payment,
          │                                 │                     returns, shipping, promotions,
          │                                 │                     customer_service, warranty, privacy}
          ▼                                 ▼                                 ▼
   data_store.py (pandas sobre data/*.csv)                         policies.py (data/policies.md,
   disponibilidade, promoções ativas, preço PIX,                   uma seção por tema)
   verificação do dono do pedido (LGPD), prazos
```

Uma mensagem, de ponta a ponta: *"Quanto custa o Takamine GD20?"* → o modelo chama
`search_products(query="Takamine GD20")` → o pandas devolve preço R$ 2.199,00, estoque 5, sem promoção
ativa, PIX R$ 2.089,05 → o modelo escreve a resposta com esses dados.

```
src/emporio/
  data_store.py   carrega os CSVs; regras de negócio sobre os dados (Python puro, com testes)
  policies.py     entrega seções do manual por tema
  tools.py        as 3 ferramentas que o LLM enxerga (docstring + type hints = schema)
  prompts.py      persona, regras de uso das ferramentas, guardrails, data/hora atual
  agent.py        monta o agente e executa um turno de conversa
  ui_format.py    escapa "$" para o markdown do Streamlit
app.py            interface de chat em Streamlit
scripts/run_examples.py
data/             dados do desafio + policies.md (transcrição do PDF em Markdown)
tests/            testes unitários + testes E2E (API real)
examples/         5 conversas geradas
```

---

## 3. Decisões técnicas

### Framework do agente — LangChain `create_agent` (LangGraph)
O `create_agent` entrega um **laço ReAct** pronto: o modelo raciocina, chama uma ferramenta, lê o
resultado e repete até conseguir responder. Ele roda sobre o LangGraph, então a memória da conversa é
só um *checkpointer*. O agente inteiro tem ~30 linhas ([`agent.py`](src/emporio/agent.py)). A
alternativa era escrever o laço à mão com o SDK do Gemini: mais controle, porém mais código para
errar, sem ganho aqui.

### Recuperação — function calling sobre dados estruturados, sem banco vetorial
- **Dados (CSVs) → ferramentas tipadas, não um agente de SQL.** As perguntas dos clientes são um
  conjunto pequeno e conhecido (buscar produtos, ver preço, rastrear pedido). Ferramentas fixas mantêm
  as regras de negócio em Python testado (só promoções ativas, PIX não acumula, disponibilidade) e
  impedem o modelo de ler dados de outros clientes, o que um agente de SQL livre poderia fazer.
- **Políticas (PDF) → uma ferramenta, uma seção por tema.** O manual tem 8 páginas e 10 seções
  numeradas. Um banco vetorial (embeddings + busca por similaridade) traria dependências e o risco de
  buscar o trecho errado, sem benefício nesse tamanho. Em vez disso, `get_policy(topic)` recebe um tema
  de uma lista fixa e devolve aquela seção. Toda consulta de política fica visível nos logs/na UI.
  **Com uma base de documentos maior, eu passaria para RAG com embeddings.**

### Modelo e provedor — Gemini `gemini-3.5-flash-lite` (tier gratuito)
O desafio permite modelos gratuitos, e o Gemini tem tool calling nativo. Comecei com o
`gemini-3.6-flash`, mas o tier gratuito dele permite só **20 requisições/dia** por modelo (cada
mensagem do cliente gasta 2–3), e a cota acabou durante os testes. As cotas gratuitas são por modelo,
então troquei para o `gemini-3.5-flash-lite` e rodei a mesma suíte E2E: todos os testes passaram, e ele
é ~4× mais rápido. O modelo é configurável por `GEMINI_MODEL`. O lite precisou de regras de prompt mais
explícitas (veja os testes de regressão) — é o trade-off de um modelo menor.

### Estratégia de prompt
[`prompts.py`](src/emporio/prompts.py):
- **Persona** tirada do manual (seção 7.1): informal mas profissional, "um amigo que entende de música".
- **Regras explícitas de roteamento**: nunca informar preço/estoque/promoção sem `search_products`;
  dados de pedido só via `check_order`; regras da loja só do `get_policy`, nunca inventadas.
- **Regras de negócio que o modelo aplica**: mostrar preço original + desconto + preço final; sugerir
  alternativas para itens esgotados; nunca confirmar devolução antes de verificar o pedido.
- **Data e hora atuais de Campo Grande** (UTC-4) injetadas no prompt, para responder "vocês estão
  abertos agora?".
- **Guardrails**: escopo (só instrumentos, nada de assuntos aleatórios), LGPD, ignorar tentativas de
  mudar as regras.
- As instruções estão em inglês; o agente responde no idioma do cliente (português por padrão, inglês
  se o cliente escrever em inglês).

### Histórico da conversa — em memória, por sessão
O `InMemorySaver` guarda o histórico por `thread_id` (um por sessão do navegador), então perguntas de
continuação como "e no PIX?" funcionam. Ele se perde quando o servidor reinicia, o que é aceitável num
protótipo. Em produção eu usaria um *checkpointer* persistente (ex.: Postgres).

### Interface — Streamlit
Um chat em ~90 linhas de Python. Um botão na barra lateral mostra as chamadas de ferramenta e seus
resultados: uma **visão de depuração** para o avaliador ver as decisões do agente. Num canal real
para clientes, ela ficaria escondida.

### Tratamento dos dados
- **PDF de políticas → Markdown, uma vez.** O `pypdf` extraiu uma palavra por linha e achatou todas as
  tabelas (horários, pagamento, frete). Transcrevi o texto fielmente para
  [`data/policies.md`](data/policies.md), mantendo as tabelas, para o modelo ler "Sábado | 09:00 às
  13:00" corretamente.
- **Promoções:** só `is_active = 1` vale (política 7.3: nunca prometer desconto vencido).
- **Disponibilidade:** `active` + estoque > 0. A navegação só mostra esses; a busca por nome mostra
  todos com o status, para o agente dizer "esgotado"/"descontinuado" em vez de "não temos".
- **Busca:** sem diferenciar maiúsculas e acentos, e também por características do campo `specs`
  ("aço" → cordas de aço, "elétrico" → tem captação).
- **PIX:** 5% sobre o preço de tabela, `null` quando há promoção (política 6.2: não acumula).
- **Prazos calculados no código, não no modelo.** O `check_order` devolve cada janela de troca e
  garantia (arrependimento de 7 dias, troca por defeito de 30 dias, garantia legal de 90 dias, garantia
  do fabricante de 6 a 24 meses) com a data final e se ainda está aberta. O modelo lite errou conta de
  datas nos testes, então vale o mesmo princípio dos preços: **o LLM explica, o Python calcula.**

---

## 4. Suposições (os dados e o manual nem sempre são consistentes)

| Situação | Suposição |
|---|---|
| O manual traz dois WhatsApps: (67) 3341-4444 (seção 1.2) e (67) 3321-4500 (seção 7) | Telefone = 3341-4444 (capa e dados da empresa); WhatsApp = 3321-4500 (seção do canal WhatsApp). |
| O e-mail aparece como `contato@empóriodamusica.com.br` (com acento) e sem acento na capa | Uso a versão sem acento, um domínio válido. |
| Os pedidos não têm a data real de recebimento | `estimated_delivery` é usado como data de recebimento para os 7 dias de arrependimento e os 90 de garantia, e o agente diz qual data usou. |
| Consultar um pedido expõe dados pessoais | O cliente precisa informar o número do pedido **e** o e-mail ou telefone da compra. Pedido inexistente e contato errado devolvem a mesma mensagem, para não ser possível descobrir números de pedido. |
| "Sugerir lojas parceiras" (seção 7.1), mas nenhuma parceira é citada | O agente sugere uma loja de acessórios de forma genérica e nunca inventa nomes. |
| O produto 130 está `coming_soon` mas aparece num pedido entregue; o total do pedido 3 (R$ 3.450) ≠ 2 × R$ 1.749 | Mantido como está: o agente lê a disponibilidade da tabela de produtos e o total da tabela de pedidos. |

---

## 5. Conversas de exemplo

Geradas com [`scripts/run_examples.py`](scripts/run_examples.py); cada uma mostra as chamadas de ferramenta (🔧).

| Arquivo | Cenário |
|---|---|
| [01_catalog_search.md](examples/01_catalog_search.md) | Violões até R$ 1000 e depois indicação de corda de aço |
| [02_price_and_payment.md](examples/02_price_and_payment.md) | Preço do Takamine GD20, PIX e 12x (dados + política) |
| [03_store_info.md](examples/03_store_info.md) | Endereço e horário de sábado |
| [04_return_request_non_trivial.md](examples/04_return_request_non_trivial.md) | **Não trivial:** devolução por arrependimento → verificação de identidade → dados do pedido + prazos de troca e garantia |
| [05_out_of_scope_and_stock.md](examples/05_out_of_scope_and_stock.md) | Cliente em inglês, acessórios, pergunta fora do assunto, item esgotado com alternativas |

---

## 6. Testes

- **Unitários** (`pytest -m "not llm"`): regras de negócio sem LLM — disponibilidade, promoções
  ativas, PIX, busca sem acento, verificação de pedido, prazos, seções de política, schemas das
  ferramentas.
- **Ponta a ponta** (`pytest -m llm`): chamadas reais ao Gemini pela mesma função `chat()` que a UI
  usa. Verificam comportamento (qual ferramenta foi chamada, fatos-chave na resposta), não o texto exato.
- **Regressão** (`test_regressão_*`): todo bug encontrado nos testes ganhou um teste que falhava antes
  da correção — ex.: prometer devolução antes de ver o pedido, responder em português a cliente em
  inglês, preço em R$ renderizado como LaTeX, busca por aço sem um produto, conta errada de garantia.

---

## 7. Limitações conhecidas e o que eu faria com mais tempo

- **Avaliação:** um conjunto maior de perguntas com respostas esperadas (*evals*) rodando a cada
  mudança, em vez de alguns testes E2E.
- **Reclamações e transbordo:** o agente diz que a reclamação será encaminhada, mas não há sistema de
  chamados; eu adicionaria uma ferramenta `create_ticket` e transbordo para humano.
- **Persistência:** o histórico fica em memória; usar um *checkpointer* persistente e registrar o
  consentimento (LGPD).
- **Limites do tier gratuito:** cotas por dia e por minuto; em produção, tier pago com fallback.
- **Recuperação de políticas:** passar para RAG com embeddings se a base de documentos crescer.
- **Canal:** integrar com a API do WhatsApp Business em vez de uma UI web.
- **Cotação de frete:** o manual diz que o valor depende de CEP e peso; não há dados para isso, então
  o agente não consegue cotar frete fora de Campo Grande.

---

## 8. Uso de assistentes de código

Usei o **Claude Code** (agente de terminal da Anthropic, modelo Claude Opus) como par de programação
durante todo o projeto:

- **Entendimento e planejamento:** ele leu comigo o desafio, os CSVs e o PDF de políticas e listou as
  pegadinhas dos dados (telefones inconsistentes, promoções vencidas, produtos sem estoque e
  descontinuados, categorias vazias). Depois passamos pelas decisões técnicas **uma de cada vez** —
  framework, recuperação das políticas, interface, fluxo de Git, idioma — com os trade-offs, e **eu
  tomei cada decisão**.
- **Implementação:** ele escreveu a maior parte do código sob minha direção, em passos pequenos, com
  testes primeiro para as regras de negócio. Revisei cada passo e rodei a aplicação eu mesmo.
- **Depuração pelo produto:** testei na UI do Streamlit e mandei prints; foi assim que achamos o bug
  do "R$" virando LaTeX e uma regra da loja inventada. Cada bug ganhou um teste de regressão antes da
  correção.
- **Checagem de fatos:** afirmações sobre bibliotecas foram conferidas no código-fonte instalado e na
  documentação oficial (ex.: a variável `GOOGLE_API_KEY`, a assinatura do `create_agent`, os modelos
  Gemini atuais).

O que aprendi e sei explicar sozinho: o laço ReAct, function calling e schemas de ferramentas, por que
não usei banco vetorial aqui e como cada regra de negócio é garantida.
