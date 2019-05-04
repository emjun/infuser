from itertools import count
from typing import Mapping

_fresh_typename_counter = count(0)


class AbstractType:
    def __eq__(self, other):
        raise NotImplementedError()


# TODO: There must be a better name than `SymbolicAbtractType`
class SymbolicAbstractType(AbstractType):
    """A simple abstract monotype."""

    def __init__(self):
        super().__init__()
        self.typename = "T" + str(next(_fresh_typename_counter))

    def __hash__(self):
        return hash((self.typename, id(self)))

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return self.typename


class DataFrameType(AbstractType):
    column_types: Mapping[str, AbstractType]

    def __init__(self, column_types):
        super().__init__()
        self.column_types = column_types

    def reassign_column_type(self, col_name: str, col_type: AbstractType) \
            -> "DataFrameType":
        new_cols = dict(self.column_types)
        new_cols[col_name] = col_type
        return DataFrameType(new_cols)

    def __hash__(self):
        return hash(tuple(t for c, t in self.column_types.items()))

    def __eq__(self, other):
        if not isinstance(other, DataFrameType):
            return False
        return dict(self.column_types) == dict(other.column_types)

    def __str__(self):
        coltyps = self.column_types
        return "DF[" + ', '.join(f"{k}: {v}" for k, v in coltyps.items()) + "]"


class PandasModuleType(AbstractType):
    def __eq__(self, other):
        return isinstance(other, PandasModuleType)


class DataFrameClsType(AbstractType):
    def __eq__(self, other):
        return isinstance(other, DataFrameClsType)
