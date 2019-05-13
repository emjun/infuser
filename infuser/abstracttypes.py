from dataclasses import dataclass
from itertools import count, chain
from typing import Sequence, Union, Set, Optional

_fresh_typename_counter = count(0)


class TypeVar:
    def __init__(self):
        super().__init__()
        self.name = "TV" + str(next(_fresh_typename_counter))

    def __hash__(self):
        return hash((self.name, id(self)))

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.name


class Type:
    @property
    def type_parameters(self) -> Sequence["Type"]:
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()


# TODO: There must be a better name than `SymbolicAbtractType`
class SymbolicAbstractType(Type):
    """A simple abstract monotype."""

    def __init__(self):
        super().__init__()
        self.typename = "t" + str(next(_fresh_typename_counter))

    @property
    def type_parameters(self) -> Sequence[Type]:
        return []

    def __hash__(self):
        return hash((self.typename, id(self)))

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.typename


@dataclass(frozen=True)
class ExtraCol:
    arg: Union[int, str]
    col_name: str
    col_type: Union[Type, TypeVar]

    def __str__(self):
        return f"{self.arg}[{self.col_name}]:{self.col_type}"


@dataclass(eq=True, frozen=True)
class CallableType(Type):
    "Type for functions and other callables. The only parametric type we have."

    arg_types: Sequence[Union[Type, TypeVar]]
    return_type: Optional[Union[Type, TypeVar]]
    "Either the return type of the function or `None` if void/unit."

    extra_cols: Set[ExtraCol]
    "Extra column-types to introduce on arguments"

    @property
    def type_parameters(self) -> Sequence[Type]:
        return list(chain(self.arg_types, [self.return_type]))

    def __str__(self):
        a = ", ".join(str(x) for x in self.arg_types)
        b = f"({a}) → {self.return_type}"
        if len(self.extra_cols):
            e = ", ".join(str(c) for c in self.extra_cols)
            b += f" \\\\ ({e})"
        return b


class PandasModuleType(Type):
    @property
    def type_parameters(self) -> Sequence[Type]:
        return []

    def __eq__(self, other):
        return isinstance(other, PandasModuleType)


class DataFrameClsType(Type):
    @property
    def type_parameters(self) -> Sequence[Type]:
        return []

    def __eq__(self, other):
        return isinstance(other, DataFrameClsType)
