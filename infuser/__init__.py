import ast
import symtable
from typing import IO

from .rules import WalkRulesVisitor
from .unification import unify


def analysis_main(client: IO[str]):
    client.seek(0)
    code_str = client.read()

    table = symtable.symtable(code_str, client.name, 'exec')
    client_ast = ast.parse(code_str, client.name)
    visitor = WalkRulesVisitor(table)
    visitor.visit(client_ast)

    visitor.type_environment
    visitor.type_constraints

    type_mapping = unify(visitor.type_constraints)
    print(type_mapping)


    raise NotImplementedError("analysis_main is incomplete")
