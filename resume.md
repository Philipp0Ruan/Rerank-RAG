# Resumo de desenvolvimento — Rerank RAG

Atualizado em 29/10/2026.

## Objetivo

Responder às cinco perguntas sobre Os Sertões combinando busca por similaridade com uma segunda seleção de trechos feita por um reranker.

Repositório: https://github.com/Philipp0Ruan/Rerank-RAG

Este projeto tem seu próprio código, notebook, dados, resultados e checklist. Pode ser desenvolvido e executado dentro desta pasta, sem importar código de outro repositório.

## Situação atual

A preparação inicial está concluída. O arquivo Python apresenta o roteiro e lê as cinco perguntas; o notebook importa essa base e também lê as perguntas. A recuperação, a geração com LLM e a avaliação final ainda precisam ser implementadas. Ainda não há respostas geradas nem resultados do experimento.

## Arquivos para continuar o trabalho

- [rerank_rag.py](rerank_rag.py): base Python da implementação.
- [notebooks/rerank_rag.ipynb](notebooks/rerank_rag.ipynb): execução das etapas no Jupyter.
- [evaluation/perguntas.json](evaluation/perguntas.json): perguntas Q1 a Q5.
- [CHECKLIST_EXECUCAO.md](CHECKLIST_EXECUCAO.md): controle local detalhado das tarefas, ignorado pelo Git.
- [README.md](README.md): apresentação do projeto e instruções de uso.
- `data/`: PDF e dados extraídos, mantidos localmente.
- `storage/`: índices e dados persistidos, a criar durante a implementação.
- `results/`: respostas e avaliações, a produzir durante a execução.

## Fluxo desta abordagem

```text
PDF → trechos → embeddings → busca de candidatos → reranking → seleção final → contexto → resposta com fontes
```

O notebook deve permitir conferir a ordem antes e depois do reranking, e a geração deve usar apenas os trechos da seleção final.

## Etapas para concluir

1. Definir o ambiente, embeddings, LLM e um reranker adequado ao português.
2. Baixar o livro, registrar o hash, extrair o texto por página e conferir a limpeza.
3. Dividir o texto em trechos com identificadores e páginas e construir o índice.
4. Recuperar candidate_k candidatos por similaridade, com candidate_k maior que final_k.
5. Aplicar o reranker e registrar ordem, pontuações e tempo antes e depois da reordenação.
6. Selecionar final_k trechos e gerar respostas em português com páginas.
7. Executar as cinco perguntas e revisar a sustentação das respostas no PDF.
8. Validar Python e notebook, persistência, tempo adicional do reranking e documentação.

## Próximo passo

Começar pela seção **1. Ambiente e escolhas** do checklist. Definir bibliotecas, modelos e configuração antes de implementar a leitura e a recuperação do livro. O Python 3.12 já foi usado na verificação inicial; o ambiente virtual, as dependências de RAG e a instalação do Jupyter ainda precisam ser preparados.

Manter as funções na base Python e usá-las no notebook, com os mesmos parâmetros, para que os dois formatos produzam resultados equivalentes.

## Perguntas que precisam ser respondidas

- **Q1:** Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?
- **Q2:** Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?
- **Q3:** Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?
- **Q4:** Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?
- **Q5:** Quais são os principais aspectos da crítica social e política presentes em "Os Sertões"? Como esses aspectos refletem a visão do autor sobre o Brasil da época?

## Critérios de conclusão

O projeto estará concluído quando o Python e o notebook executarem o pipeline completo, as cinco respostas estiverem registradas com trechos e páginas verificáveis, o índice puder ser recarregado e o README explicar uma execução reproduzível.

Registrar configurações, tempos e tokens/custo quando disponíveis. Apontar evidências insuficientes e limitações. Separar o texto de Euclides dos materiais editoriais do PDF e distinguir a página física do PDF da página impressa.

Uma comparação futura poderá usar os resultados dos outros experimentos. Isso não exige uma pasta de projeto geral nem dependências entre os repositórios.

## Fonte do livro

[PDF de Os Sertões](https://fundar.org.br/wp-content/uploads/2021/06/os-sertoes.pdf).

Ao terminar cada etapa, atualizar este resumo com a situação real e marcar no checklist apenas o que tiver sido verificado.
