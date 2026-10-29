# Rerank RAG — Os Sertões

Recupera candidatos por similaridade e usa um CrossEncoder para reordená-los antes de montar o contexto do LLM.

Implementação em [rerank_rag.py](rerank_rag.py), com base de extração e geração em [rag_base.py](rag_base.py). O [notebook](notebooks/rerank_rag.ipynb) usa as mesmas funções. Este repositório é independente dos outros experimentos.

## Situação atual

O pipeline está implementado. A validação atual usa apenas documentos, vetores e respostas fictícias, sem chamar a OpenAI. O notebook começa no modo simulado. A execução com o livro, os modelos reais e a geração da OpenAI fica pendente da aprovação do usuário.

## Recuperação

São recuperados até 24 candidatos por cosseno; o reranker seleciona até oito. O modelo é `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, na revisão `1427fd652930e4ba29e8149678df786c240d8825`. Os relatórios mostram IDs, posições e pontuações antes/depois, além do tempo de reranking. Pares são limitados a 512 tokens pelo reranker; entradas longas podem ser truncadas.

Os embeddings reais usam `intfloat/multilingual-e5-small`, revisão `614241f622f53c4eeff9890bdc4f31cfecc418b3`, com prefixos `passage: ` e `query: `. O contexto usa trechos inteiros até 16.000 caracteres. Similaridade e pontuação de reranking não são probabilidades de acerto.

## Python e Jupyter

Use Python 3.12. Para apenas simulação e notebook:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-simulated.txt
.\.venv\Scripts\python.exe rerank_rag.py simular
.\.venv\Scripts\python.exe -m jupyterlab notebooks/rerank_rag.ipynb
```

Depois da aprovação, instale as dependências reais:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe rerank_rag.py indexar
.\.venv\Scripts\python.exe rerank_rag.py avaliar
```

Configure `OPENAI_API_KEY` apenas no `.env`; nunca no notebook ou no Git. O modelo de geração inicial é `gpt-5-mini`, configurável por `OPENAI_MODEL`. A chave e os dados não são compartilhados com os outros projetos.

Os comandos `buscar "pergunta"`, `responder "pergunta"`, `avaliar --somente-busca` e `indexar --refazer` também estão disponíveis. No notebook, os modos são `simulado`, `somente_busca` e `real`.

## Corpus e avaliação

As cinco perguntas estão em [evaluation/perguntas.json](evaluation/perguntas.json). A edição do PDF é identificada por SHA-256 no [config.json](config.json); o corpus real inclui páginas físicas 33 a 653, da nota preliminar às notas de Euclides à segunda edição. A página impressa corresponde à página PDF menos 30.

Apresentação, prefácio, sumário e créditos externos ao corpus são excluídos. As notas de rodapé permanecem; notas editoriais `(PRP)` devem ser distinguidas do texto do autor. A extração pode conservar marcas de capitulares e ligaturas.

Resultados são exportados localmente em JSON e Markdown com fontes, páginas, configuração e tempos. Uso de tokens vem da API quando disponível; custo fica `null`. O verificador identifica páginas citadas fora do contexto, mas a sustentação das afirmações exige revisão humana.

A simulação usa um índice temporário, bloqueia rede e cliente OpenAI real e não lê credenciais. Seus relatórios têm `simulado: true`, `api_real_executada: false` e respostas marcadas como fictícias. Ela valida integração; não mede qualidade semântica nem desempenho de modelos reais.

O andamento e as pendências ficam em [resume.md](resume.md). O checklist de execução e os testes são apenas locais.

## Referências

- [PDF da obra](https://fundar.org.br/wp-content/uploads/2021/06/os-sertoes.pdf).
- [Desafio da Rocketseat](https://github.com/rocketseat-education/ia-para-devs/tree/master/Desafio%20RAG).
- [Modelo de embeddings](https://huggingface.co/intfloat/multilingual-e5-small).

## Docker

Dentro deste repositório, execute o cenário simulado:

```powershell
docker compose build rag
docker compose run --rm rag
```

Para abrir o Jupyter:

```powershell
docker compose up notebook
```

O Jupyter fica em http://localhost:8890; use o token informado no terminal. O notebook começa com `MODO = "simulado"`. A simulação não precisa de chave, PDF nem download de modelos. Os relatórios são salvos localmente em `results/`.

A imagem padrão usa o alvo leve `simulation`. O alvo `runtime` instala também os modelos locais via dependências PyTorch/Sentence Transformers; o download dos pesos acontece na primeira indexação. Depois da aprovação para os testes reais:

```powershell
Copy-Item .env.example .env
# Preencha a chave somente no .env local.
docker compose --profile real build real
docker compose --profile real run --rm real python rerank_rag.py indexar
docker compose --profile real run --rm real
```

Mantenha o `.env` existente se ele já estiver configurado. A imagem não inclui credenciais, dados locais, índices, testes ou relatórios. As pastas de dados, cache, índice e resultados são montadas como volumes na execução real.

O Jupyter padrão usa a imagem simulada. Para um notebook com modelos reais, depois da aprovação:

```powershell
docker compose --profile real run --rm -p 127.0.0.1:8890:8888 -v ./notebooks:/app/notebooks real python -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.root_dir=/app
```

## Versionamento

Os testes e relatórios ficam apenas no computador, ignorados pelo Git e pelo contexto Docker. O comando `simular` faz parte da aplicação e permite conferir o fluxo a partir de um clone novo. Os commits novos usam prefixos como `feat:`, `fix:`, `chore:` e `docs:`.
