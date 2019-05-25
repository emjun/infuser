import ast
from collections import defaultdict
from typing import Iterable, Mapping, Union, Dict, FrozenSet, Tuple, Set

from .abstracttypes import Type, SymbolicAbstractType, TypeVar
from .rules import TypeEqConstraint


class CannotUnifyException(Exception):
    pass


def unify_simple(constraints: Iterable[TypeEqConstraint]) -> Mapping[
    Type, Type]:
    """Create a mapping from the original to unified abstract types.

    These new abstract types should satisfy the equality constraints
    embodied by `matches`.

    This is a simplified interface to `unify` which doesn't return provenance
    details.
    """
    return unify(constraints)[0]


def unify(constraints: Iterable[TypeEqConstraint]) \
        -> Tuple[Mapping[Type, Type], Mapping[Type, FrozenSet[ast.AST]]]:
    """Create a mapping from the original to unified abstract types.

    These new abstract types should satisfy the equality constraints
    embodied by `matches`.
    """
    stack = list(constraints)
    substitutions: Dict[Union[Type, TypeVar], Union[Type, TypeVar]] = {}
    merged_sources: Dict[Type, Set[ast.AST]] = defaultdict(set)

    def substitute_everywhere(old: Union[Type, TypeVar],
                              new: Union[Type, TypeVar]) -> None:
        new_stack = []
        for c in stack:
            new_left, new_right = c.left, c.right
            if new_left == old:
                new_left = new
            if new_right == old:
                new_right = new
            new_constraint = TypeEqConstraint(new_left, new_right, c.src_node)
            new_stack.append(new_constraint)

        new_substitutions = {}
        for a, b in substitutions.items():
            new_a, new_b = a, b
            if a == old:
                new_a = new
            if b == old:
                new_b = new
            assert new_a not in new_substitutions
            new_substitutions[new_a] = new_b

        stack.clear()
        stack.extend(new_stack)
        substitutions.clear()
        substitutions.update(new_substitutions)

    while len(stack):
        top = stack.pop()
        if top.left == top.right:
            continue
        elif isinstance(top.left, SymbolicAbstractType):
            substitute_everywhere(top.left, top.right)
            substitutions[top.left] = top.right
            merged_sources[top.right].add(top.src_node)
            merged_sources[top.right].update(merged_sources[top.left])
            merged_sources[top.left] = merged_sources[top.right]
        elif isinstance(top.right, SymbolicAbstractType):
            substitute_everywhere(top.right, top.left)
            substitutions[top.right] = top.left
            merged_sources[top.left].add(top.src_node)
            merged_sources[top.left].update(merged_sources[top.right])
            merged_sources[top.right] = merged_sources[top.left]
        elif type(top.left) == type(top.right):
            left_params = top.left.type_parameters
            right_params = top.right.type_parameters
            if len(left_params) != len(right_params):
                raise CannotUnifyException()
            stack.extend(TypeEqConstraint(l, r, src_node=top.src_node)
                         for l, r in zip(left_params, right_params))
        else:
            raise CannotUnifyException()

    return substitutions, {t: frozenset(a) for t, a in merged_sources.items()}
