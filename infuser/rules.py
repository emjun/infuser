import ast
from dataclasses import dataclass
from typing import Iterable

from .abstracttypes import AbstractType, TypeReferent


@dataclass
class RuleMatch:
    """An instance of one of our inference rules matching.

    Because these always correspond to equality constraints in our analysis,
    we just need two type referents and the type they should share.
    """
    relevant_node: ast.AST
    abstract_type: AbstractType
    left: TypeReferent
    right: TypeReferent


def walk_rules(root_node: ast.AST) -> Iterable[RuleMatch]:
    raise NotImplementedError()
