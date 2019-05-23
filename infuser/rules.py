import ast
from dataclasses import dataclass
import logging
import symtable
from typing import Iterable, List, Optional, Tuple

from .abstracttypes import Type, SymbolicAbstractType, \
    PandasModuleType, DataFrameClsType, CallableType, TypeVar, ExtraCol
from .typeenv import TypingEnvironment, SymbolTypeReferant, ColumnTypeReferant

logger = logging.getLogger(__name__)


class InfuserSyntaxError(Exception):
    pass


@dataclass(eq=True, frozen=True)
class TypeEqConstraint:
    """An instance of one of our inference rules matching.

    Because these always correspond to equality constraints in our analysis,
    we just need two types about which to assert equality.

    `src_node` is a reference to the AST node which "caused" this constraint.
    Helpful for debugging.
    """
    left: Type
    right: Type
    src_node: Optional[ast.AST] = None


class WalkRulesVisitor(ast.NodeVisitor):
    type_constraints: List[TypeEqConstraint]

    _type_environment_stack: List[TypingEnvironment]
    "Map from symbols to AbstractTypes. Updated during visitation."
    # _sym_type_assignments: MutableMapping[symtable.Symbol, Type]
    # "Map from symbols to `Type`s. Updated during interpretation."

    _return_types: List[List[Type]]
    "Stack of collections of types of return expressions from `FunctionDef`s."

    def __init__(self, sym_table: symtable.SymbolTable):
        super().__init__()
        self.type_constraints = []  # we accumulate constraints for client here
        self._type_environment_stack = [TypingEnvironment()]
        self._symtable_stack: List[symtable.SymbolTable] = [sym_table]
        self._return_types = []

    @property
    def type_environment(self) -> TypingEnvironment:
        return self._type_environment_stack[0]

    def visit_Import(self, node: ast.Import):
        for a in node.names:
            if a.name == "pandas":
                sym_name = a.name if a.asname is None else a.asname
                pandas_sym = self._symtable_stack[-1].lookup(sym_name)
                self._type_environment_stack[-1][
                    SymbolTypeReferant(pandas_sym)] = PandasModuleType()
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "pandas" and node.level == 0:
            for a in node.names:
                if a.name == "DataFrame":
                    sym_name = a.name if a.asname is None else a.asname
                    sym = self._symtable_stack[-1].lookup(sym_name)
                    self._type_environment_stack[-1][sym] = DataFrameClsType()
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Remember that this may be called inside `visit_Assign` etc.

        # TODO: Add special handling for recursion, which makes side effects
        #  tricky to analyze

        self.generic_visit(node)

        if not isinstance(node.func, ast.Name):
            return

        # Look up the function in symbol table and our type env.
        func_sym = self._symtable_stack[-1].lookup(node.func.id)
        func_type: Optional[CallableType] = \
            self._type_environment_stack[-1].get(SymbolTypeReferant(func_sym))
        if func_type is None:
            raise ValueError(f"Called undefined function {node.func.id}")
        for extra_col in func_type.extra_cols:
            if isinstance(extra_col.arg, int):
                node_arg = node.args[extra_col.arg]
                # We could also extend subscripts
                if not isinstance(node_arg, ast.Name):
                    logger.warn(f"Only name arguments supported with "
                                f"constraints; got {node_arg}")
                    continue
                name_sym = self._symtable_stack[-1].lookup(node_arg.id)
                ref = ColumnTypeReferant(symbol=SymbolTypeReferant(name_sym),
                                         column_names=extra_col.col_names)
                prev_type = self._type_environment_stack[-1].get_or_bake(ref)
                self.type_constraints.append(
                    TypeEqConstraint(prev_type, extra_col.col_type, node))
            else:
                raise NotImplementedError("Only position args supported so far")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # TODO: Warn if function re-definition?

        self._push_scope(node.name)
        self._return_types.append([])
        type_env = self._type_environment_stack[-1]
        top_symtable = self._symtable_stack[-1]

        # Push args into the new type env.
        assert (isinstance(top_symtable, symtable.Function))
        arg_types: List[Tuple[TypeVar, symtable.Symbol]] = []
        for sym in top_symtable.get_symbols():
            # TODO: Make sure this is actually in argument order
            # TODO: Ensure this doesn't grab second-order nested func. params.
            if sym.is_parameter():
                arg_type_var = TypeVar()
                type_env[SymbolTypeReferant(sym)] = arg_type_var
                arg_types.append((arg_type_var, sym))

        # Avoid decorators. Manually call visit on statements.
        for child in node.body:
            self.visit(child)

        # Choose an arbitrary return type for the signature
        rt = None
        if len(self._return_types[-1]):
            rt = self._return_types[-1][0]

        # Add side effects to `extra_cols`
        extra_cols = set()
        for i, (a, sym) in enumerate(arg_types):
            c = type_env.subscripted_name_closure(SymbolTypeReferant(sym))
            for sub, t in c.items():
                extra_cols.add(ExtraCol(i, sub.column_names, t))

        # TODO: Add global re-assignments to `extra_constaints`

        func_type = CallableType(
            tuple([x[0] for x in arg_types]), rt, frozenset(extra_cols))

        self._return_types.pop()
        self._pop_scope()
        self._type_environment_stack[-1][SymbolTypeReferant(
            self._symtable_stack[-1].lookup(node.name))] = func_type
        logger.debug(f"Assigned function {node.name}: {func_type}")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if isinstance(node.target, (ast.Name, ast.Subscript)):
                tgt_type = self._get_expr_type(node.target)
                val_type = self._get_expr_type(node.value)
                if val_type is not None:
                    constraint = TypeEqConstraint(tgt_type, val_type, node)
                    self.type_constraints.append(constraint)
                else:
                    logger.debug("Skipping AugAssign from value type "
                                 "%s", type(node.value))
            elif isinstance(node.target, ast.Attribute):
                logger.debug("Skipping augmented assignment to attribute")
        else:
            logger.debug("Ignoring AugAssign op type %s", type(node.op))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        type_env = self._type_environment_stack[-1]

        self.generic_visit(node)

        if isinstance(node.value, ast.Call) and isinstance(
                self._get_expr_type(node.value.func), DataFrameClsType):
            # TODO: Implement DataFrame calls
            call = node.value
            possible_data_asts = [k for k in call.keywords if k.arg == "data"]
            if len(possible_data_asts) == 0:
                raise NotImplementedError("DataFrame not given `data`")
            elif len(possible_data_asts) >= 2:
                raise Exception("DataFrame given `data` twice")
            data_ast = possible_data_asts[0].value
            if not isinstance(data_ast, ast.Dict):
                raise NotImplementedError("Non-dict `data` unsupported")
            for k, v in zip(data_ast.keys, data_ast.values):
                # Assign a subscript-extended reference to a fresh type for
                # every target and `data` key literal
                if k is None:
                    raise NotImplementedError("dict `**` unsupported")
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        t_sym_ref = SymbolTypeReferant(
                            self._symtable_stack[-1].lookup(t.id))
                        type_env[t_sym_ref.add_subscript(k.s)] = \
                            SymbolicAbstractType()
                    else:
                        raise NotImplementedError(
                            "Only name targets are implemented")
        else:
            value_type = self._get_expr_type(node.value, bake_fresh=True)

            for t in node.targets:
                if isinstance(t, ast.Name):
                    # Copy type and all its subscripts
                    t_sym_ref = SymbolTypeReferant(
                        self._symtable_stack[-1].lookup(t.id))
                    type_env[t_sym_ref] = value_type
                    for r, ta in type_env.subscripted_name_closure(
                            t_sym_ref).items():
                        type_env[r.replace_symbol(t_sym_ref)] = ta
                elif isinstance(t, ast.Subscript):
                    root, subscript_chain = accum_string_subscripts(t)
                    if not isinstance(root, ast.Name):
                        return None
                    ref = ColumnTypeReferant(
                        symbol=SymbolTypeReferant(
                            self._symtable_stack[-1].lookup(root.id)),
                        column_names=tuple(subscript_chain))
                    type_env[ref] = value_type
                else:
                    raise NotImplementedError(
                        "Only name and subscripted targets are implemented")

    def visit_Compare(self, node: ast.Compare) -> None:
        self.generic_visit(node)

        # Map these comparisons into some triples because `ast.Compare` has a
        # very, very odd design
        cmp_triples = [(node.left, node.ops[0], node.comparators[0])]
        if len(node.ops) >= 2:
            cmp_triples += \
                [(l, o, r) for l, o, r in
                 zip(node.comparators, node.ops[1:], node.comparators[1:])]

        for l, o, r in cmp_triples:
            l_type = self._get_expr_type(l)
            r_type = self._get_expr_type(r)
            if l_type is not None and r_type is not None:
                self.type_constraints.append(
                    TypeEqConstraint(l_type, r_type, node))

    def _get_expr_type(self, expr: ast.expr, bake_fresh=False) \
            -> Optional[Type]:
        """Return a type assigned to `expr` or `None` if irrelevant to analysis

        Note that this requires the `_symtable_stack` to be set so that `expr`
        in the latest symbol table. Only meant to be called during AST
        visitation.
        """
        # TODO: Look up by symbol in the case of variable references

        if isinstance(expr, ast.Name):
            if not isinstance(expr.ctx, ast.Load):
                raise ValueError(f"Looked up name {expr} with non-Load ctx")
            sym = SymbolTypeReferant(self._symtable_stack[-1].lookup(expr.id))
            to_r = self._type_environment_stack[-1].get(sym)
            if to_r is None and bake_fresh:
                to_r = self._type_environment_stack[-1].get_or_bake(sym)
            return to_r

        if isinstance(expr, ast.Attribute):
            value_type = self._get_expr_type(expr.value)
            if isinstance(value_type, PandasModuleType):
                if expr.attr == "DataFrame":
                    return DataFrameClsType()
            return None

        if isinstance(expr, ast.Subscript):
            # The only subscripts we care about are chains of string literals.
            root, subscript_chain = accum_string_subscripts(expr)
            if not isinstance(root, ast.Name):
                return None
            ref = ColumnTypeReferant(
                symbol=SymbolTypeReferant(
                    self._symtable_stack[-1].lookup(root.id)),
                column_names=tuple(subscript_chain))
            # TODO: Return fresh variable if none and `bake_fresh`
            if bake_fresh:
                return self._type_environment_stack[-1].get_or_bake(ref)
            return self._type_environment_stack[-1].get(ref)

        if isinstance(expr, (ast.Str, ast.JoinedStr, ast.Num)):
            # A perpetually fresh type
            return SymbolicAbstractType()

        if isinstance(expr, ast.Call):
            raise ValueError("_get_expr_type shouldn't be used on calls")
        if isinstance(expr, ast.Attribute):
            raise NotImplementedError("Attributes not implemented")

        raise NotImplementedError(
            f"Unable to judge type for expression: {expr}")

    def _push_scope(self, name: str) -> None:
        # Push a freshly baked, empty type environment
        self._type_environment_stack.append(TypingEnvironment())

        # Find and push the relevant symbol table
        for child in self._symtable_stack[-1].get_children():
            if child.get_name() == name:
                self._symtable_stack.append(child)
                return
        raise ValueError(f"Couldn't find child symbol table for {name}")

    def _pop_scope(self) -> None:
        self._symtable_stack.pop()
        self._type_environment_stack.pop()


def accum_string_subscripts(expr: ast.Subscript) -> Tuple[ast.AST, List[str]]:
    subs: List[str] = []
    while isinstance(expr, ast.Subscript) \
            and isinstance(expr.slice, ast.Index) \
            and isinstance(expr.slice.value, ast.Str):
        subs.insert(0, expr.slice.value.s)
        expr = expr.value
    return expr, subs


def walk_rules(root_node: ast.AST, table: symtable.SymbolTable) \
        -> Iterable[TypeEqConstraint]:
    visitor = WalkRulesVisitor(table)
    visitor.visit(root_node)
    return visitor.type_constraints
