import ast
from itertools import combinations
import logging
import symtable
import sys
from typing import IO, TypeVar

from .rules import WalkRulesVisitor, STAGES
from .unification import unify

T = TypeVar("T")
V = TypeVar("V")

logger = logging.getLogger(__name__)


def analysis_main(client: IO[str]):
    client.seek(0)
    code_str = client.read()

    table = symtable.symtable(code_str, client.name, 'exec')
    client_ast = ast.parse(code_str, client.name)
    visitor = WalkRulesVisitor(table)
    visitor.visit(client_ast)

    type_env = visitor.type_environment
    new_envs = []
    for stage in STAGES:
        subs = unify(
            visitor.type_constraints[None] | visitor.type_constraints[stage])
        new_envs.append(type_env.substitute_types(subs))

    for (env_a, stg_a), (env_b, stg_b) in combinations(zip(new_envs, STAGES),
                                                       2):
        for name in set(env_a.keys()) & set(env_b.keys()):
            if env_a[name] != env_b[name]:
                print(f"{stg_a} and {stg_b} disagree about {name}",
                      file=sys.stderr)
