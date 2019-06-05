from dataclasses import dataclass
from itertools import count, chain
from typing import Sequence, Union, FrozenSet, MutableMapping, Mapping, Tuple, \
    TypeVar, Iterable, List

T = TypeVar("T")
V = TypeVar("V")

MonomorphingCache = MutableMapping["TypeVar", "Type"]
MonomorphizeReturn = \
    Tuple["Type", MutableMapping[Union["TypeVar", "Type"], "Type"]]
_fresh_typename_counter = count(0)


def merge_disjoint_maps(srcs: Iterable[Mapping[T, V]]) -> MutableMapping[T, V]:
    to_r = {}
    for src in srcs:
        for k, v in src.items():
            if k in to_r:
                raise ValueError(f"Key {k} duplicated")
            to_r[k] = v
    return to_r


class TypeVar:
    def __init__(self):
        super().__init__()
        self.name = "TV" + str(next(_fresh_typename_counter))

    @property
    def type_parameters(self) -> Sequence[Union["TypeVar", "Type"]]:
        return []

    def replace_type(self, old: Union["Type", "TypeVar"],
                     new: Union["Type", "TypeVar"]) \
            -> Union["Type", "TypeVar"]:
        if old == self:
            return new
        return self

    def make_monomorphic(self, cache: MonomorphingCache) -> MonomorphizeReturn:
        if self not in cache:
            cache[self] = SymbolicAbstractType()
        return cache[self], {self: cache[self]}

    def __hash__(self):
        return hash((self.name, id(self)))

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.name


class Type:
    @property
    def type_parameters(self) -> Sequence[Union[TypeVar, "Type"]]:
        raise NotImplementedError()

    def make_monomorphic(self, cache: MonomorphingCache) -> MonomorphizeReturn:
        """Copy the type, replacing type variables with monotypes.
        """
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def replace_type(self, old: Union["Type", TypeVar],
                     new: Union["Type", TypeVar]) -> Union["Type", TypeVar]:
        raise NotImplementedError()


class UnitType(Type):
    def make_monomorphic(self, cache: MonomorphingCache) -> MonomorphizeReturn:
        return self, {}

    @property
    def type_parameters(self) -> Sequence[Union[TypeVar, "Type"]]:
        return []

    def replace_type(self, old: Union["Type", TypeVar],
                     new: Union["Type", TypeVar]) -> Union["Type", TypeVar]:
        if isinstance(old, UnitType):
            return new
        return self

    def __hash__(self):
        return 9571321

    def __eq__(self, other):
        return isinstance(other, UnitType)

    def __str__(self):
        return "()"

    def __repr__(self):
        return "UnitType()"


# TODO: There must be a better name than `SymbolicAbstractType`
class SymbolicAbstractType(Type):
    """A simple abstract monotype."""

    def __init__(self):
        super().__init__()
        self.typename = "t" + str(next(_fresh_typename_counter))

    def make_monomorphic(self, cache: MonomorphingCache) -> MonomorphizeReturn:
        return self, {}

    @property
    def type_parameters(self) -> Sequence[Union[TypeVar, Type]]:
        return []

    def replace_type(self, old: Union[Type, TypeVar],
                     new: Union[Type, TypeVar]) -> Union[Type, TypeVar]:
        if old == self:
            return new
        return self

    def __hash__(self):
        return hash((self.typename, id(self)))

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.typename

    # Add this to print the type when debugging
    # def __repr__(self):
    #     return self.typename


@dataclass(eq=True, frozen=True)
class TupleType(Type):
    """Param. poly. type for Python tuples."""
    element_types: Tuple[Union[Type, TypeVar], ...]

    @property
    def type_parameters(self) -> Sequence[Union[TypeVar, "Type"]]:
        return self.element_types

    def make_monomorphic(self, cache: MonomorphingCache) -> MonomorphizeReturn:
        buf: List[MonomorphizeReturn] = \
            [e.make_monomorphic(cache) for e in self.element_types]
        m = TupleType(tuple(a for a, _ in buf))
        n = merge_disjoint_maps(b for _, b in buf)
        return m, n

    def replace_type(self, old: Union[Type, TypeVar],
                     new: Union[Type, TypeVar]) -> Union[Type, TypeVar]:
        if old == self:
            return new
        return TupleType(tuple(e.replace_type(old, new)
                               for e in self.element_types))

    def __str__(self):
        inner = ",".join(str(x) for x in self.element_types)
        return f"<{inner}>"


@dataclass(frozen=True)
class ExtraCol:
    arg: Union[int, str]
    col_names: Tuple[str, ...]
    col_type: Union[Type, TypeVar]

    def make_monomorphic(self, cache: MonomorphingCache) \
            -> Tuple["ExtraCol", Mapping[Union[Type, TypeVar], Type]]:
        m, subs = self.col_type.make_monomorphic(cache)
        n = ExtraCol(self.arg, self.col_names, m)
        return n, subs

    def __str__(self):
        return f"{self.arg}[{self.col_names}]:{self.col_type}"


@dataclass(eq=True, frozen=True)
class CallableType(Type):
    "Type for functions and other callables. The only parametric type we have."

    param_types: Tuple[Union[Type, TypeVar], ...]
    return_type: Union[Type, TypeVar]
    "Either the return type of the function or `None` if void/unit."

    extra_cols: FrozenSet[ExtraCol]
    "Extra column-types to introduce on arguments"

    @property
    def type_parameters(self) -> Sequence[Union[TypeVar, Type]]:
        return list(chain(self.param_types, [self.return_type]))

    def make_monomorphic(self, cache: MonomorphingCache) \
            -> Tuple["CallableType", Mapping[Union[TypeVar, Type], Type]]:
        params_mono: List[MonomorphizeReturn] = \
            [a.make_monomorphic(cache) for a in self.param_types]
        return_mono, return_subs = self.return_type.make_monomorphic(cache)
        extras_mono = [a.make_monomorphic(cache) for a in self.extra_cols]
        n = CallableType(
            param_types=tuple(m for m, _ in params_mono),
            return_type=return_mono,
            extra_cols=frozenset(m for m, _ in extras_mono))
        subs = merge_disjoint_maps([return_subs] +
                                   [s for _, s in params_mono] +
                                   [s for _, s in extras_mono])
        return n, subs

    def replace_type(self, old: Union[Type, TypeVar],
                     new: Union[Type, TypeVar]) -> Union[Type, TypeVar]:
        if old == self:
            return new
        new_rt = self.return_type.replace_type(old, new)
        return CallableType(
            param_types=tuple(a.replace_type(old, new) for a in self.param_types),
            return_type=new_rt,
            extra_cols=self.extra_cols)

    def __str__(self):
        a = ", ".join(str(x) for x in self.param_types)
        b = f"({a}) → {self.return_type}"
        if len(self.extra_cols):
            e = ", ".join(str(c) for c in self.extra_cols)
            b += f" \\\\ ({e})"
        return b


class SingletonType(Type):
    @property
    def type_parameters(self) -> Sequence[Union[TypeVar, Type]]:
        return []

    def replace_type(self, old: Union[Type, TypeVar],
                     new: Union[Type, TypeVar]) -> Union[Type, TypeVar]:
        if old == self:
            return new
        return self

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(str(type(self)))


class PandasModuleType(SingletonType):
    pass


class ScipyModuleType(SingletonType):
    pass


class StatsmodelsModuleType(SingletonType):
    pass


class ScipyStatsModuleType(SingletonType):
    pass


class MannWhitneyUFunc(SingletonType):
    pass


class DataFrameClsType(Type):
    @property
    def type_parameters(self) -> Sequence[Union[TypeVar, Type]]:
        return []

    def replace_type(self, old: Union[Type, TypeVar],
                     new: Union[Type, TypeVar]) -> Union[Type, TypeVar]:
        if old == self:
            return new
        return self

    def __eq__(self, other):
        return isinstance(other, DataFrameClsType)

    def __hash__(self):
        return 31
