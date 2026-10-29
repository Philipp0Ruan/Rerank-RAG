# Rerank RAG — Os Sertões

Este projeto vai responder perguntas sobre **Os Sertões** usando duas etapas de seleção de contexto. Primeiro, a busca por similaridade encontra trechos candidatos. Depois, um modelo de reranking reavalia a relação de cada trecho com a pergunta e reorganiza os resultados.

Faz parte do Desafio RAG, com três abordagens em repositórios separados:
[Naive RAG](https://github.com/Philipp0Ruan/Naive-RAG),
[Parent RAG](https://github.com/Philipp0Ruan/Parent-RAG) e
[Rerank RAG](https://github.com/Philipp0Ruan/Rerank-RAG).

## Situação atual

A estrutura inicial está pronta. O notebook contém o roteiro de desenvolvimento; a leitura do PDF, a recuperação e a geração das respostas ainda serão implementadas. Ainda não há respostas geradas nem resultados de avaliação.

## Como vai funcionar

```text
PDF → trechos → embeddings → busca de candidatos → reranking → seleção final → resposta com fontes
```

- **Buscar candidatos:** Recuperar um conjunto inicial de trechos por similaridade; definir candidate_k maior que final_k.
- **Aplicar o reranking:** Escolher um reranker com suporte adequado ao português e registrar nome, versão, pontuações e tempo da reordenação.
- **Selecionar o contexto final:** Reordenar os candidatos pelo reranker e enviar somente os final_k selecionados, com fontes e páginas. Exibir a ordem antes e depois.

Vamos usar a mesma divisão de texto e os mesmos embeddings do Naive RAG. Assim será possível avaliar o efeito do reranking, incluindo o tempo extra de processamento.

## Perguntas do desafio

1. Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?
2. Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?
3. Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?
4. Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?
5. Quais são os principais aspectos da crítica social e política presentes em "Os Sertões"? Como esses aspectos refletem a visão do autor sobre o Brasil da época?

As perguntas também estão em [evaluation/perguntas.json](evaluation/perguntas.json), com identificadores iguais nos três projetos.

## Organização

```text
notebooks/rerank_rag.ipynb   # roteiro do experimento
evaluation/perguntas.json     # cinco perguntas do desafio
data/                        # PDF e dados extraídos, somente locais
results/                     # respostas e avaliações, quando forem geradas
```

O notebook fica em [notebooks/rerank_rag.ipynb](notebooks/rerank_rag.ipynb).

## Desenvolvimento e execução

Vamos implementar o notebook por etapas: preparar o ambiente, carregar e conferir o livro, construir a recuperação, gerar as respostas e avaliar as evidências.

Python e Jupyter serão usados para o experimento. O provedor do LLM, os modelos, as bibliotecas e suas versões serão definidos na implementação. As instruções de instalação e execução serão adicionadas aqui depois de validadas. Por enquanto, o notebook é um roteiro e não executa um pipeline RAG.

## Como avaliar

O notebook deve mostrar candidatos, ordem e pontuações antes/depois do reranking e gerar a resposta com os trechos da seleção final.

Para cada pergunta, vamos guardar a resposta, os trechos utilizados, as páginas, os parâmetros e o tempo de execução. A revisão vai conferir se a resposta atende à pergunta e se as fontes sustentam o que o modelo escreveu. Afirmações sem evidência devem ser identificadas, e o modelo deve indicar quando o contexto for insuficiente.

Na comparação entre os projetos, vamos manter o mesmo PDF, as mesmas perguntas e, sempre que possível, o mesmo modelo de geração, embeddings, prompt e limite de contexto. As diferenças de configuração serão registradas junto dos resultados.

## Material de referência

- [PDF de Os Sertões disponibilizado pela Fundar](https://fundar.org.br/wp-content/uploads/2021/06/os-sertoes.pdf).
- [Solução de referência da Rocketseat](https://github.com/rocketseat-education/ia-para-devs/tree/master/Desafio%20RAG).

O PDF será salvo localmente em `data/os-sertoes.pdf`. A edição inclui material editorial; vamos separar esse conteúdo do texto de Euclides antes de avaliar as respostas e distinguir o número da página no PDF da numeração impressa no livro.
