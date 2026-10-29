"""Rerank RAG: busca vetorial seguida de CrossEncoder."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import time

import numpy as np

from rag_base import Config, RAIZ, RAGBase


@dataclass(frozen=True)
class RerankConfig(Config):
    candidate_k: int = 24
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_revision: str = "1427fd652930e4ba29e8149678df786c240d8825"

    def __post_init__(self):
        super().__post_init__()
        if self.candidate_k <= self.top_k:
            raise ValueError("candidate_k deve ser maior que top_k.")


class RerankRAG(RAGBase):
    abordagem = "rerank"

    def __init__(self, root: Path = RAIZ, config: RerankConfig | None = None):
        super().__init__(root, config or RerankConfig.carregar(root))

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.config.reranker_model,
                            revision=self.config.reranker_revision,
                            cache_folder=str(self.root / ".cache" / "reranker"),
                            device="cpu", max_length=512, trust_remote_code=False)

    def buscar(self, pergunta, top_k=None):
        k = self.config.top_k if top_k is None else top_k
        if not 1 <= k < self.config.candidate_k:
            raise ValueError("top_k deve ser positivo e menor que candidate_k.")
        candidatos = super().buscar(pergunta, self.config.candidate_k)
        inicio = time.perf_counter()
        if candidatos:
            scores = np.asarray(self.reranker.predict(
                [(pergunta, item["text"]) for item in candidatos],
                show_progress_bar=False), dtype=float).reshape(-1)
            if len(scores) != len(candidatos) or not np.isfinite(scores).all():
                raise ValueError("O reranker devolveu pontuações inválidas.")
            ordem = np.argsort(-scores, kind="stable")
            resultado = [{**candidatos[i], "rerank_score": float(scores[i]),
                          "initial_rank": int(i) + 1, "final_rank": pos + 1}
                         for pos, i in enumerate(ordem[:k])]
        else:
            resultado = []
        self.ultima_busca = {
            "antes": [{"id": c["id"], "similarity": c["similarity"], "rank": i + 1}
                      for i, c in enumerate(candidatos)],
            "depois": [{"id": c["id"], "rerank_score": c["rerank_score"],
                        "initial_rank": c["initial_rank"], "final_rank": c["final_rank"]}
                       for c in resultado],
            "tempo_reranking_s": round(time.perf_counter() - inicio, 4),
            "modelo": self.config.reranker_model,
            "revision": self.config.reranker_revision,
        }
        return resultado


if __name__ == "__main__":
    from cli import executar
    raise SystemExit(executar(RerankRAG))
