import unittest

from infuser.abstracttypes import SymbolicAbstractType, CallableType
from infuser.rules import TypeEqConstraint
from infuser.unification import unify


class TestUnification(unittest.TestCase):
    # TODO(Lukas): Add more tests

    def test_unify_with_empty_matches_list(self):
        self.assertEqual(len(unify([])), 0)

    def test_unify_with_empty_matches_generator(self):
        g = iter([])
        self.assertEqual(len(unify(g)), 0)

    def test_unify_with_simple_transitivity(self):
        a, b, c = [SymbolicAbstractType() for _ in range(3)]
        constraints = [
            TypeEqConstraint(a, b, src_node=None),
            TypeEqConstraint(b, c, src_node=None)]
        substitutions = unify(constraints)
        self.assertEqual(2, len(substitutions))
        self.assertEqual(2, len(set(substitutions.keys())))
        self.assertEqual(1, len(set(substitutions.values())))

    def test_unify_with_parameterized_types(self):
        a, b, c = [SymbolicAbstractType() for _ in range(3)]
        c1 = CallableType(arg_types=(a, b), return_type=None,
                          extra_cols=frozenset())
        c2 = CallableType(arg_types=(a, c), return_type=None,
                          extra_cols=frozenset())
        substitutions = unify([TypeEqConstraint(c1, c2, src_node=None)])
        self.assertTrue((substitutions == {c: b}) or (substitutions == {b: c}))
