"""Simulação offline com dados e respostas explicitamente fictícios."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rag_base import (RAIZ, carregar_perguntas, gerar_resposta, ler_json,
                      salvar_json, salvar_markdown, sha256)
from rerank_rag import RerankConfig, RerankRAG

AVISO = "SIMULAÇÃO: documentos, vetores e respostas fictícios; nenhuma chamada à OpenAI."


class TokenizadorSimulado:
    def encode(self, texto):
        return texto.split()

    def __call__(self, texto, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", texto)]}


class ModeloSimulado:
    max_seq_length = 512
    tokenizer = TokenizadorSimulado()

    def __init__(self, perguntas):
        self.perguntas = {"query: " + p["pergunta"]: i for i, p in enumerate(perguntas)}

    def encode(self, textos, **kwargs):
        resultado = []
        for texto in textos:
            vetor = np.zeros(7, dtype=np.float32)
            if texto in self.perguntas:
                vetor[self.perguntas[texto]] = 1
            elif texto.startswith("passage: ") and re.search(r"TEMA_(\d+)", texto):
                tema = int(re.search(r"TEMA_(\d+)", texto).group(1))
                vetor[tema] = 0.8 if "PREFERIDO" in texto else 1.0
                if "PREFERIDO" in texto:
                    vetor[5] = 0.6
            else:
                vetor[6] = 1
            resultado.append(vetor)
        return np.array(resultado)


class RerankerSimulado:
    def predict(self, pares, **kwargs):
        return np.array([10.0 if "PREFERIDO" in texto else 1.0 for _, texto in pares])


class ClienteSimulado:
    def __init__(self):
        self.responses = self
        self.chamadas = []

    def create(self, **argumentos):
        self.chamadas.append(argumentos)
        pagina = re.search(r"\[PDF p\. (\d+);", argumentos["input"]).group(1)
        texto = ("SIMULAÇÃO — saída programada para validar a integração "
                 f"[PDF p. {pagina}]. Não é uma resposta real sobre Os Sertões.")
        return SimpleNamespace(status="completed", output_text=texto, model="llm-simulado-v1",
                               id=f"simulado-{len(self.chamadas)}", usage=None)


def preparar_cenario(root, perguntas, base):
    paginas = []
    for i in range(5):
        for variante in ("CANDIDATO", "PREFERIDO"):
            numero = len(paginas) + 1
            paginas.append({"id": f"p{numero}", "pdf_page": numero, "printed_page": numero,
                            "source": "fixture://rerank",
                            "text": f"TEMA_{i} {variante}. Documento sintético sem conteúdo histórico real."})
    digest = hashlib.sha256(json.dumps(paginas).encode()).hexdigest()
    config = replace(base, pdf_url="fixture://corpus-sintetico", pdf_sha256=digest,
                     first_pdf_page=1, last_pdf_page=len(paginas), printed_page_offset=0,
                     embedding_model="embeddings-simulados", embedding_revision="fixture-v1",
                     top_k=2, min_similarity=0.5, candidate_k=6, reranker_model="reranker-simulado", reranker_revision="fixture-v1")
    rag = RerankRAG(root, config)
    rag.__dict__["modelo"] = ModeloSimulado(perguntas)
    rag.__dict__["reranker"] = RerankerSimulado()
    trechos = rag.dividir_documentos(paginas)
    vetores = rag.modelo.encode(["passage: " + t["text"] for t in trechos])
    storage = root / "storage"
    storage.mkdir(parents=True)
    np.save(storage / "embeddings.npy", vetores, allow_pickle=False)
    manifesto = {"simulado": True, "assinatura": config.assinatura(),
                 "dimensoes": vetores.shape[1], "total_trechos": len(trechos),
                 "config": asdict(config), "embeddings_sha256": sha256(storage / "embeddings.npy")}
    salvar_json(storage / "indice.json", {"manifesto": manifesto, "trechos": trechos,
                                          "extras": rag.extras_indice()})
    salvar_json(root / "evaluation/perguntas.json", {"perguntas": perguntas})
    rag.carregar_indice()
    return rag


def executar_simulacao(root=RAIZ, nome="simulacao"):
    if not re.fullmatch(r"[\w.-]+", nome) or nome in (".", ".."):
        raise ValueError("Nome de relatório inválido.")
    root = Path(root)
    perguntas = carregar_perguntas(root)
    if len(perguntas) != 5:
        raise ValueError("O cenário espera cinco perguntas.")
    cliente = ClienteSimulado()

    def gerar(pergunta, contexto, config, pasta):
        return gerar_resposta(pergunta, contexto, config, pasta, client=cliente)

    bloqueio = AssertionError("Rede e cliente OpenAI real bloqueados na simulação.")
    with TemporaryDirectory() as pasta:
        rag = preparar_cenario(Path(pasta), perguntas, RerankConfig.carregar(root))
        with patch("socket.socket.connect", side_effect=bloqueio), \
             patch("socket.socket.connect_ex", side_effect=bloqueio), \
             patch("socket.create_connection", side_effect=bloqueio), \
             patch("openai.OpenAI", side_effect=bloqueio), \
             patch("rag_base.configuracao_openai", return_value={"model": "llm-simulado-v1", "temperature": None}), \
             patch("rag_base.gerar_resposta", side_effect=gerar):
            caminho = rag.avaliar(nome="lote")
            negativo = rag.responder("Pergunta sem correspondência no corpus sintético.")
        dados = ler_json(caminho)
        if len(cliente.chamadas) != 5 or negativo["geracao_executada"]:
            raise AssertionError("Cenário simulado inconsistente.")
    dados.update(simulado=True, aviso=AVISO, api_real_executada=False,
                 origem_dados="corpus sintético e vetores programados",
                 revisao_humana="não avalia qualidade de LLM real",
                 chamadas_simuladas=5, cenario_sem_evidencia=negativo)
    for item in [*dados["resultados"], negativo]:
        item.update(modo="simulado", simulado=True, geracao_simulada=item["geracao_executada"],
                    geracao_executada=False, api_real_executada=False, uso_tokens=None, custo=None)
        item["citacoes"]["revisao_semantica"] = "não avaliada: cenário sintético"
    destino = root / "results" / f"{nome}.json"
    salvar_json(destino, dados)
    salvar_markdown(destino.with_suffix(".md"), dados)
    return destino
