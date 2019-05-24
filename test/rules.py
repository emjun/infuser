import ast
from itertools import combinations
import symtable
from typing import Iterable
import unittest

from infuser import unify
from infuser.abstracttypes import TypeVar, Type
from infuser.rules import WalkRulesVisitor, TypeEqConstraint
from infuser.typeenv import TypingEnvironment, ColumnTypeReferant, \
    SymbolTypeReferant


def types_connected(origin, destination,
                    env: TypingEnvironment,
                    constraints: Iterable[TypeEqConstraint]) -> bool:
    # graph = defaultdict(set)
    # for constraint in constraints:
    #     graph[constraint.left].add(constraint.right)
    #     graph[constraint.right].add(constraint.left)
    #
    # visited, queue = set(), [origin]
    # while queue:
    #     vertex = queue.pop(0)
    #     if vertex == destination:
    #         return True
    #     if vertex not in visited:
    #         visited.add(vertex)
    #         queue.extend(graph[vertex] - visited)
    # return False

    if origin == destination:
        return True
    substitutions = unify(constraints)
    if origin in substitutions and destination in substitutions and \
            substitutions[origin] == substitutions[destination]:
        return True
    elif origin == substitutions.get(destination):
        return True
    elif substitutions.get(origin) == destination:
        return True
    return False


class TestRules(unittest.TestCase):

    # TODO: Make sure to test column re-assignment
    # TODO: Make sure to test whole DataFrame re-assignment

    def test_simple_assignments(self):
        examples = [(2, "x = 1; y = x"),
                    (3, "x = 2; y = x; z = y"),
                    (4, "x = 3; y = x; a = x; z = y")]
        for expected_type_cnt, code_str in examples:
            root_node = ast.parse(code_str, '<unknown>', 'exec')
            table = symtable.symtable(code_str, '<unknown>', 'exec')
            visitor = WalkRulesVisitor(table)
            visitor.visit(root_node)

            # There should be two things in the typing environment, and they
            # should be or be constrained to be equal
            self.assertEqual(expected_type_cnt, len(visitor.type_environment))
            for a, b in combinations(visitor.type_environment.values(), 2):
                self.assertTrue(self.types_connected(a, b, visitor))

    def test_simple_comparisons(self):
        CMPS = ["==", "!=", ">", "<", ">=", "<="]
        examples = [(2, f"x = 1; y = 2; x {op} y") for op in CMPS]
        examples += [(3, f"x = 1; y = 2; z = 'hi'; x {op} y {op} z")
                     for op in CMPS]

        for expected_type_cnt, code_str in examples:
            root_node = ast.parse(code_str, '<unknown>', 'exec')
            table = symtable.symtable(code_str, '<unknown>', 'exec')
            visitor = WalkRulesVisitor(table)
            visitor.visit(root_node)

            self.assertEqual(expected_type_cnt, len(visitor.type_environment))
            for a, b in combinations(visitor.type_environment.values(), 2):
                self.assertTrue(self.types_connected(a, b, visitor))

    def test_tuple_with_intermediate_pseudo_erasure(self):
        # Please enjoy this test name of nonsense words.
        code_str = "x = 1; y = 2; a = (x, y); b = a; i, j = b"

        root_node = ast.parse(code_str, '<unknown>', 'exec')
        table = symtable.symtable(code_str, '<unknown>', 'exec')

        # Get the symbols corresponding to `x`, `y`, `i`, and `j`
        x_s, y_s, i_s, j_s = [table.lookup(n) for n in "xyij"]

        visitor = WalkRulesVisitor(table)
        visitor.visit(root_node)

        x_t, y_t, i_t, j_t = [visitor.type_environment[SymbolTypeReferant(s)]
                              for s in (x_s, y_s, i_s, j_s)]
        self.assertTrue(self.types_connected(i_t, x_t, visitor))
        self.assertTrue(self.types_connected(j_t, y_t, visitor))

    def test_subscripts_are_assigned_properly_during_destructuring(self):
        code_str = "x = {}; y = {}; x['Hi'] = 1; y['Bye'] = x['Hi']; y = {}"
        root_node = ast.parse(code_str, '<unknown>', 'exec')
        table = symtable.symtable(code_str, '<unknown>', 'exec')
        visitor = WalkRulesVisitor(table)
        visitor.visit(root_node)

        # There should be four things in the typing environment and none of
        # them should have the same type
        self.assertEqual(3, len(visitor.type_environment))
        for a, b in combinations(visitor.type_environment.values(), 2):
            self.assertFalse(self.types_connected(a, b, visitor))


    def test_types_abandoned_on_reassign(self):
        code_str = "x = 2; y = x; y = 2"
        root_node = ast.parse(code_str, '<unknown>', 'exec')
        table = symtable.symtable(code_str, '<unknown>', 'exec')
        visitor = WalkRulesVisitor(table)
        visitor.visit(root_node)

        # There should be two types -- `x` and `y` -- and they should be
        # disconnected at the end of iteration, despite having interacted
        # before `y` was re-assigned
        self.assertEqual(2, len(visitor.type_environment))
        for a, b in combinations(visitor.type_environment.values(), 2):
            self.assertFalse(self.types_connected(a, b, visitor))

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

    def test_type_eq_after_augmented_assignment(self):
        # Equivalent examples
        examples = [
            "import pandas as pd\n"
            "df = pd.DataFrame(data={'Price': [10.], 'Price2': [20.]})\n"
            f"df['Price2'] {op_str} df['Price']""" for op_str in ("+=", "-=")]

        for code_str in examples:
            root_node = ast.parse(code_str, '<unknown>', 'exec')
            table = symtable.symtable(code_str, '<unknown>', 'exec')

            # Get the sub-exprs corresponding to the augmented assignment
            assignment_stmts = [n for n in ast.walk(root_node)
                                if isinstance(n, ast.AugAssign)]
            assert len(assignment_stmts) == 1
            df_symbol_ref = SymbolTypeReferant(table.lookup("df"))
            df_price_ref = ColumnTypeReferant(df_symbol_ref, ("Price",))
            df_price2_ref = ColumnTypeReferant(df_symbol_ref, ("Price2",))

            # Make sure the two columns on `df` are unified
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
