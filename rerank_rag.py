"""Preparação do Rerank RAG para o desafio com Os Sertões.

Este arquivo guarda o roteiro e lê as perguntas locais. A recuperação e a
geração serão implementadas nas próximas etapas. O notebook importa esta base.
"""

from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
TITULO = 'Rerank RAG'
STATUS = "Preparação inicial: o pipeline RAG ainda será implementado."

# As etapas seguem a mesma ordem do notebook.
ROTEIRO: tuple[tuple[str, str], ...] = (
    ('1. Ambiente e configuração',
     'Definir e instalar as dependências, selecionar modelos, carregar variáveis de ambiente e registrar versões e parâmetros. Não inserir chaves no notebook.'),
    ('2. PDF e corpus',
     'Baixar o PDF para `data/os-sertoes.pdf`, registrar seu hash e extrair o texto com fonte e páginas. Conferir a limpeza e separar os textos editoriais do corpus da obra.\n\nA execução deve localizar a raiz do repositório independentemente da pasta em que o Jupyter foi aberto.'),
    ('3. Buscar candidatos',
     'A implementar: Recuperar um conjunto inicial de trechos por similaridade; definir candidate_k maior que final_k.'),
    ('4. Aplicar o reranking',
     'A implementar: Escolher um reranker com suporte adequado ao português e registrar nome, versão, pontuações e tempo da reordenação.'),
    ('5. Selecionar o contexto final',
     'A implementar: Reordenar os candidatos pelo reranker e enviar somente os final_k selecionados, com fontes e páginas. Exibir a ordem antes e depois.'),
    ('6. Prompt e geração',
     'Montar o contexto com fontes e páginas. Responder em português, distinguir interpretações do autor de fatos atuais e indicar quando faltarem evidências. Usar o mesmo prompt base na comparação entre abordagens.'),
    ('7. Perguntas do desafio',
     'Carregar `evaluation/perguntas.json` e executar as cinco perguntas, preservando os identificadores.\n\n- **Q1:** Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?\n- **Q2:** Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?\n- **Q3:** Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?\n- **Q4:** Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?\n- **Q5:** Quais são os principais aspectos da crítica social e política presentes em "Os Sertões"? Como esses aspectos refletem a visão do autor sobre o Brasil da época?'),
    ('8. Avaliação e resultados',
     'O notebook deve mostrar candidatos, ordem e pontuações antes/depois do reranking e gerar a resposta com os trechos da seleção final.\n\nGuardar respostas, evidências, páginas, parâmetros, latência e uso de tokens quando disponível em `results/`. Conferir manualmente a sustentação das respostas no PDF.\n\nVamos usar a mesma divisão de texto e os mesmos embeddings do Naive RAG. Assim será possível avaliar o efeito do reranking, incluindo o tempo extra de processamento.'),
    ('9. Reprodução e conclusão',
     'Reiniciar o kernel, executar todas as células em ordem e validar a recarga do índice. Documentar instalação e execução no README somente depois de verificar o fluxo completo. Registrar resultados reais e limitações.'),
)


def carregar_perguntas() -> list[dict[str, str]]:
    """Lê as cinco perguntas, independentemente da pasta de execução."""
    caminho = RAIZ / "evaluation" / "perguntas.json"
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return dados["perguntas"]


def mostrar_preparacao() -> None:
    """Mostra o andamento e o roteiro, sem executar o pipeline futuro."""
    print(f"{TITULO} — Os Sertões")
    print(STATUS)
    print("\nEtapas:")
    for titulo, _ in ROTEIRO:
        print(f"  {titulo}")
    print("\nPerguntas do desafio:")
    for item in carregar_perguntas():
        print(f"  {item['id']}: {item['pergunta']}")


if __name__ == "__main__":
    mostrar_preparacao()
