"""Comandos do experimento."""

import argparse
import json
import sys

from openai import APIError
from rag_base import configuracao_openai


def executar(classe):
    parser = argparse.ArgumentParser(description=classe.__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)
    indice = comandos.add_parser("indexar")
    indice.add_argument("--refazer", action="store_true")
    for nome in ("buscar", "responder"):
        comando = comandos.add_parser(nome)
        comando.add_argument("pergunta")
        if nome == "buscar":
            comando.add_argument("--k", type=int)
    avaliar = comandos.add_parser("avaliar")
    avaliar.add_argument("--somente-busca", action="store_true")
    avaliar.add_argument("--nome")
    simular = comandos.add_parser("simular")
    simular.add_argument("--nome", default="simulacao")
    args = parser.parse_args()
    try:
        if args.comando == "simular":
            from simulacao import AVISO, executar_simulacao
            print(AVISO)
            print(executar_simulacao(nome=args.nome))
            return 0
        rag = classe()
        if args.comando == "indexar":
            resultado = rag.indexar(args.refazer)
        elif args.comando == "buscar":
            resultado = rag.buscar(args.pergunta, args.k)
        elif args.comando == "responder":
            configuracao_openai(rag.root)
            resultado = rag.responder(args.pergunta)
        else:
            print(rag.avaliar(args.somente_busca, args.nome))
            return 0
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        return 0
    except APIError as erro:
        print(f"Falha na OpenAI: HTTP {getattr(erro, 'status_code', None) or 'indisponível'}.", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
