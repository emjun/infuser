from dataclasses import dataclass
from symtable import Symbol
from typing import Mapping, Union, Tuple, TypeVar, MutableMapping

from .abstracttypes import Type, SymbolicAbstractType


# TODO: Make sure we're using "type signature" correctly


class TypeReferant:
    pass


@dataclass(eq=True, frozen=True)
class SymbolTypeReferant(TypeReferant):
    symbol: Symbol

    def add_subscript(self, str_sub: str) -> "ColumnTypeReferant":
        return ColumnTypeReferant(symbol=self, column_names=(str_sub,))


@dataclass(eq=True, frozen=True)
class ColumnTypeReferant(TypeReferant):
    symbol: SymbolTypeReferant
    column_names: Tuple[str]

    def add_subscript(self, str_sub: str) -> "ColumnTypeReferant":
        return ColumnTypeReferant(
            symbol=self.symbol,
            column_names=self.column_names + tuple(str_sub))

    def replace_symbol(self, new: Union[SymbolTypeReferant, Symbol]) \
            -> "ColumnTypeReferant":
        if isinstance(new, Symbol):
            return self.replace_symbol(SymbolTypeReferant(new))
        return ColumnTypeReferant(symbol=new, column_names=self.column_names)


class TypingEnvironment(dict,
                        MutableMapping[TypeReferant, Union[Type, TypeVar]]):

    def __setitem__(self, key: TypeReferant, value: Union[Type, TypeVar]) \
            -> None:
        assert isinstance(key, TypeReferant), f"key type was {type(key)}"
        super().__setitem__(key, value)

    def get_or_bake(self, k: TypeReferant) -> Union[Type, TypeVar]:
        if k not in self:
            self[k] = SymbolicAbstractType()
        return self[k]

    def subscripted_name_closure(self, base: TypeReferant) \
            -> Mapping[ColumnTypeReferant, Type]:
        if isinstance(base, SymbolTypeReferant):
            return {k: v for k, v in self.items()
                    if isinstance(k, ColumnTypeReferant) and k.symbol == base}
        else:
            raise NotImplementedError("Non-symbolic bases not implemented")
