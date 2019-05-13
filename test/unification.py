import unittest

from infuser.abstracttypes import SymbolicAbstractType, DataFrameType
from infuser.rules import TypeEqConstraint
from infuser.unification import unify


class TestUnification(unittest.TestCase):
    # TODO(Lukas): Add more tests

    def test_unify_with_empty_matches_list(self):
        self.assertEqual(len(unify([])), 0)

    def test_unify_with_empty_matches_generator(self):
        g = iter([])
        self.assertEqual(len(unify(g)), 0)
    
    def small_example(self):
        
        at_1 = SymbolicAbstractType()
        at_2 = SymbolicAbstractType()
        rm = TypeEqConstraint()
        rm.left = at_1
        rm.right = at_2
        
        
        at_3 = SymbolicAbstractType()
        rm1 = TypeEqConstraint()
        rm1.left = at_3
        rm1.right = at_1
        
        at_4 = SymbolicAbstractType()
        at_5 = SymbolicAbstractType()
        rm2 = TypeEqConstraint()
        rm2.left = at_4
        rm2.right = at_5
    
        matches = [rm, rm1, rm2]
        mapping = unify(matches)
        
        for x, y in mapping.items():
            print(x, end=" "),
            print(y)
        
    
    def df_example(self):
        at_1 = SymbolicAbstractType()
        at_2 = SymbolicAbstractType()
        rm = TypeEqConstraint()
        rm.left = at_1
        rm.right = at_2
        
        column_types = {'1' : at_1}
        dt_1 = DataFrameType(column_types)
        dt_2 = DataFrameType(column_types)

        rm1 = TypeEqConstraint()
        rm1.left = dt_1
        rm1.right = dt_2
        
        matches = [rm, rm1]
        
        mapping = unify(matches)
        for x, y in mapping.items():
            print(x, end=" "),
            print(y)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        