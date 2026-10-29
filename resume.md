# Resumo de desenvolvimento — Rerank RAG

Atualizado em 29/10/2026.

## Estado

Implementação disponível em Python e Jupyter Notebook, com execução simulada por padrão. O repositório tem Dockerfile e Compose para aplicação e notebook. Os testes, checklists e relatórios permanecem locais e ignorados pelo Git.

São recuperados até 24 candidatos por cosseno; o reranker seleciona até oito. O modelo é `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, na revisão `1427fd652930e4ba29e8149678df786c240d8825`. Os relatórios mostram IDs, posições e pontuações antes/depois, além do tempo de reranking. Pares são limitados a 512 tokens pelo reranker; entradas longas podem ser truncadas.

## Verificação

A validação desta etapa é simulada: usa documentos sintéticos, embeddings programados e cliente falso de LLM, sem chamadas à OpenAI. A simulação cobre Q1 a Q5 e um cenário sem evidência. Os resultados fictícios não demonstram a qualidade de respostas sobre o livro.

As imagens simulation dos três projetos foram construídas. Neste repositório, CLI e notebook executaram dentro do Docker sem rede, com os cenários simulados. O notebook publicado fica sem saídas; a cópia executada e os relatórios permanecem em results/, localmente. Os testes locais passaram (8 casos). O alvo leve `simulation` não baixa pesos; o alvo `runtime` inclui dependências para modelos reais. A execução real permanece pendente, conforme a orientação do usuário.

## Como continuar

1. Rodar `docker compose run --rm rag` para repetir a simulação.
2. Abrir `docker compose up notebook` e manter `MODO = "simulado"` enquanto não houver aprovação.
3. Depois da aprovação, configurar a chave local, construir o alvo `runtime`, indexar o livro e executar as cinco perguntas reais.
4. Revisar as evidências e citações e registrar limitações antes de concluir o desafio.

O README contém os comandos completos. Os arquivos `CHECKLIST_EXECUCAO.md`, `tests/` e `results/` ficam apenas no computador. Nenhuma execução real foi agendada.
