from dataclasses import dataclass
from symtable import Symbol
from typing import Mapping, Union, Tuple, TypeVar, MutableMapping, List

from .abstracttypes import Type, SymbolicAbstractType


# TODO: Make sure we're using "type signature" correctly


class TypeReferant:
    pass


@dataclass(eq=True, frozen=True)
class SymbolTypeReferant(TypeReferant):
    symbol: Symbol

    def __str__(self):
        return f"SymbolTypeReferant({self.symbol.get_name()})"

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

    def __contains__(self, item):
        return super().__contains__(item)

    def get_or_bake(self, k: TypeReferant) -> Union[Type, TypeVar]:
        if k not in self:
            self[k] = SymbolicAbstractType()
        return self[k]

    def remove_assignments(self, sym: SymbolTypeReferant) -> None:
        to_remove: List[TypeReferant] = [sym]
        to_remove += list(self.subscripted_name_closure(sym))
        for ref in to_remove:
            if ref in self:
                del self[ref]

    def copy_assignments(self, orig: SymbolTypeReferant,
                         novel: SymbolTypeReferant) -> None:
        "Replace `orig` and its subscripts with those of `novel`."

        # Remove all `from` and subscripts
        keys_to_remove = []
        for k, v in self.items():
            if isinstance(k, ColumnTypeReferant) and k.symbol == orig:
                keys_to_remove.append(k)
            elif k == orig:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self[k]

        # Copy `novel` and subscripts to `orig`
        for k, v in self.items():
            if isinstance(k, ColumnTypeReferant) and k.symbol == novel:
                self[k.replace_symbol(novel)] = v
            elif k == orig:
                self[novel] = v

    def subscripted_name_closure(self, base: TypeReferant) \
            -> Mapping[ColumnTypeReferant, Type]:
        if isinstance(base, SymbolTypeReferant):
            return {k: v for k, v in self.items()
                    if isinstance(k, ColumnTypeReferant) and k.symbol == base}
        else:
            raise NotImplementedError("Non-symbolic bases not implemented")
