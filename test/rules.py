import ast
import symtable
from typing import List, cast
import unittest

from infuser.rules import WalkRulesVisitor


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
            matches = list(visitor.type_constraints)
            self.assertEqual(2, len(matches))
            final_d_cols = cast(DataFrameType, visitor._sym_type_assignments[
                visitor._symtable_stack[-1].lookup("d")]).column_types
            self.assertEqual(2, len(final_d_cols))
            self.assertEqual(1, len(set(final_d_cols.values())))

    def test_walk_rules_over_simple_assignment_examples_with_funcs(self):
        # Equivalent examples
        examples = [
            """
import pandas as pd
df = pd.DataFrame(data={"Price": [100.]})
def mod(d):
  d["Price2"] = d["Price"]
mod(df)""", """
def ex():
    from pandas import DataFrame
    d = DataFrame(data={"Price": [100.]})
    d["Price2"] = d["Price"]
ex()""",
        ]

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
            matches = cast(List[ast.Expr], list(visitor.type_constraints))
            self.assertEqual(len(matches), 1)
            self.assertSetEqual({visitor._get_expr_type(matches[0].left),
                                 visitor._get_expr_type(matches[0].right)},
                                {visitor._get_expr_type(p1_expr),
                                 visitor._get_expr_type(p2_expr)})
