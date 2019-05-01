from typing import Iterable, Mapping

from abstracttypes import AbstractType
from rules import RuleMatch


def unify(matches: Iterable[RuleMatch]) -> Mapping[AbstractType, AbstractType]:
    """Create a mapping from the original to unified abstract types.

    These new abstract types should satisfy the equality constraints
    embodied by `matches`.
    """
    raise NotImplementedError()
