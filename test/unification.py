import unittest

from infuser.abstracttypes import SymbolicAbstractType, CallableType, TupleType, \
    UnitType
from infuser.rules import TypeEqConstraint
from infuser.unification import unify_simple


class TestUnification(unittest.TestCase):
    # TODO(Lukas): Add more tests

    def test_unify_with_empty_matches_list(self):
        self.assertEqual(len(unify_simple([])), 0)

    def test_unify_with_empty_matches_generator(self):
        g = iter([])
        self.assertEqual(len(unify_simple(g)), 0)

    def test_unify_with_simple_transitivity(self):
        a, b, c = [SymbolicAbstractType() for _ in range(3)]
        constraints = [
            TypeEqConstraint(a, b, src_node=None),
            TypeEqConstraint(b, c, src_node=None)]
        substitutions = unify_simple(constraints)
        self.assertEqual(2, len(substitutions))
        self.assertEqual(2, len(set(substitutions.keys())))
        self.assertEqual(1, len(set(substitutions.values())))

    def test_unify_with_parameterized_types(self):
        a, b, c = [SymbolicAbstractType() for _ in range(3)]
        c1 = CallableType(param_types=(a, b), return_type=UnitType(),
                          extra_cols=frozenset())
        c2 = CallableType(param_types=(a, c), return_type=UnitType(),
                          extra_cols=frozenset())
        substitutions = unify_simple([TypeEqConstraint(c1, c2, src_node=None)])
        self.assertTrue((substitutions == {c: b}) or (substitutions == {b: c}))

    def test_unification_of_parameterized_types_with_intervening_symbolic(self):
        a, b, c = [SymbolicAbstractType() for _ in range(3)]
        intervener = SymbolicAbstractType()
        tup1 = TupleType((a, b))
        tup2 = TupleType((c, b))
        substitutions = unify_simple([
            TypeEqConstraint(tup1, intervener, src_node=None),
            TypeEqConstraint(tup2, intervener, src_node=None)])
        self.assertTrue(substitutions.get(a) == c or substitutions.get(c) == a)
