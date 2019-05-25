import ast
import logging
import symtable
from typing import IO

from .rules import WalkRulesVisitor
from .unification import unify

logger = logging.getLogger(__name__)


def analysis_main(client: IO[str]):
    client.seek(0)
    code_str = client.read()

    table = symtable.symtable(code_str, client.name, 'exec')
    client_ast = ast.parse(code_str, client.name)
    visitor = WalkRulesVisitor(table)
    visitor.visit(client_ast)

    wrangling_subs = unify(
        visitor.type_constraints[None] | visitor.type_constraints["WRANGLING"])
    analysis_subs = unify(
        visitor.type_constraints[None] | visitor.type_constraints["ANALYSIS"])

    pass
    # TODO: Finish `analysis_main`
