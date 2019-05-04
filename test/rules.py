import ast
import symtable
import unittest

from abstracttypes import ColumnReferent
from infuser.rules import walk_rules, RuleMatch

import pandas as pd

df = pd.DataFrame(data={"Price": [100.]})


class TestRules(unittest.TestCase):

    def test_walk_rules_over_simple_assignment_example(self):
        code_str = """
import pandas as pd
df = pd.DataFrame(data={"Price": [100.]})
def mod(d):
  d["Price2"] = d["Price"]
mod(df)"""
        root_node = ast.parse(code_str, '<unknown>', 'exec')
        table = symtable.symtable(code_str, '<unknown>', 'exec')
        it = walk_rules(root_node)
        rule_matches = list(it)
        self.assertEqual(len(rule_matches), 1)
        rule_match = rule_matches[0]
        df_symbol = table.get_children()[0].lookup("d")
        self.assertEqual(rule_match,
                         RuleMatch(ColumnReferent(df_symbol, "Price2"),
                                   ColumnReferent(df_symbol, "Price")))
