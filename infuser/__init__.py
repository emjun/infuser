import ast
import symtable
from typing import IO

from .rules import WalkRulesVisitor


def analysis_main(client: IO[str]):
    client.seek(0)
    code_str = client.read()

    table = symtable.symtable(code_str, client.name, 'exec')
    client_ast = ast.parse(client.read(), client.name)
    visitor = WalkRulesVisitor(table)
    visitor.visit(client_ast)

    raise NotImplementedError("analysis_main is incomplete")
