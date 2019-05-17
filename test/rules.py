import ast
from collections import defaultdict
import symtable
from typing import Iterable
import unittest

from infuser.abstracttypes import TypeVar, Type
from infuser.rules import WalkRulesVisitor, TypeEqConstraint
from infuser.typeenv import TypingEnvironment, ColumnTypeReferant, \
    SymbolTypeReferant


def types_connected(origin, destination,
                    env: TypingEnvironment,
                    constraints: Iterable[TypeEqConstraint]) -> bool:
    graph = defaultdict(set)
    for constraint in constraints:
        graph[constraint.left].add(constraint.right)
        graph[constraint.right].add(constraint.left)

    visited, queue = set(), [origin]
    while queue:
        vertex = queue.pop(0)
        if vertex == destination:
            return True
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return False


class TestRules(unittest.TestCase):

    # TODO: Make sure to test column re-assignment
    # TODO: Make sure to test whole DataFrame re-assignment

    def test_immediate_unification_over_simple_straightline_assignment_examples(
            self):
        # Equivalent examples
        examples = [
            """
import pandas as pd
d = pd.DataFrame(data={"Price": [100.]})
d["Price2"] = d["Price"]"""]

        for code_str in examples:
            root_node = ast.parse(code_str, '<unknown>', 'exec')
            table = symtable.symtable(code_str, '<unknown>', 'exec')

            # Get the sub-exprs corresponding to the column assignment
            assignment_stmts = [n for n in ast.walk(root_node)
                                if isinstance(n, ast.Assign)]
            assert len(assignment_stmts) == 2
            p2_expr: ast.Subscript = assignment_stmts[-1].targets[0]
            p1_expr: ast.Subscript = assignment_stmts[-1].value
            assert isinstance(p2_expr, ast.Subscript)
            assert isinstance(p1_expr, ast.Subscript)

            # Assert that an equality constraint is baked for these
            # expressions. Remember types are not unified yet, so
            # we need the type judgement from the `WalkRulesVisitor`
            # specifically for the expressions `p1_expr`, `p2_expr`
            # above.
            visitor = WalkRulesVisitor(table)
            visitor.visit(root_node)

            # Make sure that `d["Price"]` and `d["Price2"]` are equal
            self.assertTrue(self.types_connected(p1_expr, p2_expr, visitor))

    def test_walk_rules_over_simple_assignment_examples_with_funcs(self):
        # Equivalent examples
        examples = [
            """
import pandas as pd
df = pd.DataFrame(data={"Price": [100.]})
def mod(d):
  d["Price2"] = d["Price"]
mod(df)"""]

        for code_str in examples:
            root_node = ast.parse(code_str, '<unknown>', 'exec')
            table = symtable.symtable(code_str, '<unknown>', 'exec')

            # Get the sub-exprs corresponding to the column assignment
            assignment_stmts = [n for n in ast.walk(root_node)
                                if isinstance(n, ast.Assign)]
            assert len(assignment_stmts) == 2
            df_symbol_ref = SymbolTypeReferant(table.lookup("df"))
            df_price_ref = ColumnTypeReferant(df_symbol_ref, ("Price",))
            df_price2_ref = ColumnTypeReferant(df_symbol_ref, ("Price2",))

            # Make sure the two columns on `df` are unified despite their
            # unification happening as the result of the side effects in
            # `mod`'s type signature.
            visitor = WalkRulesVisitor(table)
            visitor.visit(root_node)
            type_env = visitor.type_environment
            self.assertTrue(self.types_connected(type_env[df_price_ref],
                                                 type_env[df_price2_ref],
                                                 visitor))

    @staticmethod
    def types_connected(a, b, visitor: WalkRulesVisitor) -> bool:
        if not isinstance(a, (Type, TypeVar)):
            a = visitor._get_expr_type(a)
        if not isinstance(b, (Type, TypeVar)):
            b = visitor._get_expr_type(b)
        return types_connected(a, b,
                               visitor.type_environment,
                               visitor.type_constraints)
