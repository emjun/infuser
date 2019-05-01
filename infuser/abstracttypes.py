from dataclasses import dataclass
from typing import NewType

# These are just opaque objects, so lets just use the integers.
# We will, however, need to maintain and pass around some sort
# of environment so we can reliably generate sequential integers.
AbstractType = NewType("AbstractType", int)


# The below should correspond to the expressions about which
# our inference rules are asserting equality. Keep in mind
# these don't always obviously correspond to unique expressions
# in the syntax tree, e.g. DataFrame columns (a kind of Series).
class TypeReferent(dataclass):
    pass
