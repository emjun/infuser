import ast
from dataclasses import dataclass
import logging
import symtable
from typing import Iterable, List, Union, Optional, cast, \
    MutableMapping

from .abstracttypes import AbstractType, SymbolicAbstractType, \
    DataFrameType, PandasModuleType, DataFrameClsType

EQ_INDUCING_OPS = ["+", "-", "+=", "-="]

logger = logging.getLogger(__name__)


class InfuserSyntaxError(Exception):
    pass


@dataclass
class RuleMatch:
    """An instance of one of our inference rules matching.

    Because these always correspond to equality constraints in our analysis,
    we just need two types about which to assert equality.

    `src_node` is a reference to the AST node which "caused" this constraint.
    Helpful for debugging.
    """
    left: AbstractType
    right: AbstractType
    src_node: Optional[ast.AST] = None


class WalkRulesVisitor(ast.NodeVisitor):
    rule_matches: List[RuleMatch]

    _sym_type_assignments: MutableMapping[symtable.Symbol, AbstractType]
    "Map from symbols to `AbstractType`s. Updated during interpretation."

    def __init__(self, sym_table: symtable.SymbolTable):
        super().__init__()
        self.rule_matches = []
        self._sym_type_assignments = {}
        self._table_stack: List[symtable.SymbolTable] = [sym_table]

        # TODO: Track Pandas, DataFrame, and Series symbols on import
        self._pandas_symbols = set()
        self._dataframe_symbols = set()
        self._series_symbols = set()

    def visit_Import(self, node: ast.Import):
        for a in node.names:
            if a.name == "pandas":
                sym_name = a.name if a.asname is None else a.asname
                pandas_sym = self._table_stack[-1].lookup(sym_name)
                self._pandas_symbols.add(pandas_sym)
                # TODO: Store the Symbol, using `sym_name`
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == "pandas" and node.level == 0:
            for a in node.names:
                sym_name = a.name if a.asname is None else a.asname
                if a.name == "DataFrame":
                    dataframe_sym = self._table_stack[-1].lookup(sym_name)
                    self._dataframe_symbols.add(dataframe_sym)
                elif a.name == "Series":
                    series_sym = self._table_stack[-1].lookup(sym_name)
                    self._series_symbols.add(series_sym)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._enter_child_namespace(node.name)
        self.generic_visit(node)
        self._exit_child_namespace()

    def visit_Assign(self, node: ast.Assign):
        # TODO: Only care if left-hand side is interesting type?
        # TODO: Handle literal assignments?
        self.generic_visit(node)
        value_type = self.get_expr_type(node.value)
        assert value_type is not None
        for t in node.targets:
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Index) \
                    and isinstance(t.slice.value, ast.Str) \
                    and isinstance(self.get_expr_type(t.value), DataFrameType):
                t_type = cast(DataFrameType, self.get_expr_type(t.value))
                new_t_type = t_type.reassign_column_type(t.slice.value.s,
                                                         value_type)
                self._sym_type_assignments[
                    self._table_stack[-1].lookup(t.value.id)] = new_t_type
                self.rule_matches.append(
                    RuleMatch(value_type, new_t_type, src_node=node))
            elif isinstance(t, ast.Name):
                t_type = self.get_expr_type(t)
                self._sym_type_assignments[
                    self._table_stack[-1].lookup(t.id)] = value_type
                self.rule_matches.append(
                    RuleMatch(value_type, t_type, src_node=node))
            else:
                raise NotImplementedError()

    def get_expr_type(self, expr: ast.expr) -> Optional[AbstractType]:
        """Return a type assigned to `expr` or `None` if irrelevant to analysis

        Note that this requires the `_table_stack` to be set such that expr
        in the latest symbol table. Only meant to be called internally during
        AST traversal.
        """
        # TODO: Look up by symbol in the case of variable references

        if isinstance(expr, ast.Name):
            return self._get_symbol_type(expr.id)

        if isinstance(expr, ast.Attribute):
            value_type = self.get_expr_type(expr.value)
            if isinstance(value_type, PandasModuleType):
                if expr.attr == "DataFrame":
                    return DataFrameClsType()
            return None

        if isinstance(expr, ast.Call):
            # Return a type for calls to DataFrame constructor, but no other
            # function call.
            func_type = self.get_expr_type(expr.func)
            if func_type is None:
                return None
            elif isinstance(func_type, DataFrameClsType):
                if len(expr.args):
                    raise NotImplementedError("Don't yet support normal args "
                                              "to DataFrame constructor")
                args_dict = {}
                for k in expr.keywords:
                    name, val = k.arg, k.value
                    if name in args_dict:
                        raise InfuserSyntaxError(
                            f"Argument {name} specified twice")
                    args_dict[name] = val

                data = args_dict["data"]
                if isinstance(data, ast.Dict):
                    cols_to_types = {}
                    for data_key, data_val in zip(data.keys, data.values):
                        if data_key is None:
                            logger.info(
                                "We don't support 3.5-style dict expansion")
                            return None
                        elif isinstance(data_key, ast.Str):
                            cols_to_types[data_key.s] = SymbolicAbstractType()
                        else:
                            logger.info("We don't support non-strings in data")
                            return None
                    return DataFrameType(cols_to_types)
                else:
                    logger.info(
                        "We only analyze dicts given `data` in DataFrame constructor")
                    logger.debug(f"Saw data param: {data}")
                    return None

            raise NotImplementedError(
                "Should implement semantics for this call")

        if isinstance(expr, ast.Attribute):
            raise NotImplementedError()

        if isinstance(expr, ast.Subscript):
            if not isinstance(expr.ctx, ast.Load):
                return None
            value_type = self.get_expr_type(expr.value)
            if value_type is None:
                return None
            if isinstance(value_type, DataFrameType):
                assert isinstance(expr.slice, ast.Index), \
                    "Advanced indexing/slicing not implemented"
                assert isinstance(expr.slice.value, ast.Str), \
                    "Non string literal subscripting not implemented"
                return value_type.column_types[expr.slice.value.s]

        raise NotImplementedError("Unable to judge type for expression")

    def _get_symbol_type(self,
                         sym: Union[symtable.Symbol, str]) -> AbstractType:
        if isinstance(sym, str):
            return self._get_symbol_type(self._table_stack[-1].lookup(sym))

        if sym in self._pandas_symbols:
            return PandasModuleType()
        try:
            return self._sym_type_assignments[sym]
        except KeyError:
            pass
        fresh_type = SymbolicAbstractType()
        self._sym_type_assignments[sym] = fresh_type
        return fresh_type

    def _enter_child_namespace(self, name: str) -> None:
        for child in self._table_stack[-1].get_children():
            if child.get_name() == name:
                self._table_stack.append(child)
                return
        raise ValueError(f"Couldn't find child symbol table for {name}")

    def _exit_child_namespace(self) -> None:
        self._table_stack.pop()


def walk_rules(root_node: ast.AST, table: symtable.SymbolTable) \
        -> Iterable[RuleMatch]:
    visitor = WalkRulesVisitor(table)
    visitor.visit(root_node)
    return visitor.rule_matches
