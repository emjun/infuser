import unittest

from infuser.unification import unify


class TestUnification(unittest.TestCase):
    # TODO(Lukas): Add more tests

    def test_unify_with_empty_matches_list(self):
        self.assertEqual(len(unify([])), 0)

    def test_unify_with_empty_matches_generator(self):
        g = iter([])
        self.assertEqual(len(unify(g)), 0)
