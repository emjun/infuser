import ast
from itertools import combinations, combinations_with_replacement
import logging
import symtable
import sys
from typing import TypeVar, Mapping, Iterable, \
    MutableMapping, Set

from infuser.cli import CLIPrinter
from .rules import WalkRulesVisitor, STAGES
from .unification import unify

T = TypeVar("T")
V = TypeVar("V")

logger = logging.getLogger(__name__)


def merge_set_valued_mapping(sources: Iterable[Mapping[T, Iterable[V]]]) \
        -> Mapping[T, Set[V]]:
    r: MutableMapping[T, Set[V]] = {}
    for src in sources:
        for k, v in src.items():
            dest = r.setdefault(k, set())
            dest.update(v)
    return r


def analysis_main(code_str: str, filename: str, printer: CLIPrinter) -> int:
    table = symtable.symtable(code_str, filename, 'exec')
    client_ast = ast.parse(code_str, filename)
    visitor = WalkRulesVisitor(table)
    visitor.visit(client_ast)

    # Complain if no stages were found
    if sum(1 for k in visitor.type_constraints if k is not None) == 0:
        print("No top-level stages found. (Note that Infuser doesn't support "
              "stage headings inside `if __name__ == '__main__'` blocks.)",
              file=sys.stderr)
        return 1

    # Okay. Time for the real analysis
    type_env = visitor.type_environment
    new_envs = []
    src_maps = []
    for stage in STAGES:
        subs, srcs = unify(
            visitor.type_constraints[None] | visitor.type_constraints[stage])
        new_envs.append(type_env.substitute_types(subs))
        src_maps.append(srcs)

    for (e1, m1, s1), (e2, m2, s2) in \
            combinations(zip(new_envs, src_maps, STAGES), 2):
        merged_maps = merge_set_valued_mapping([m1, m2])
        all_shared_names = set(e1.keys()) & set(e2.keys())
        for n1, n2 in combinations_with_replacement(all_shared_names, 2):
            same_under_one = (e1[n1] == e1[n2])
            same_under_two = (e2[n1] == e2[n2])
            if same_under_one != same_under_two:
                # TODO: Factor out the warnings I/O for easier I/O testing
                src_nodes = (merged_maps[e1[n1]] | merged_maps[e1[n2]] |
                             merged_maps[e2[n1]] | merged_maps[e2[n2]])
                printer.warn(
                    f"Disagreement about {n1.display_str()} and {n2.display_str()}",
                    set((x.lineno - 1, x.col_offset) for x in src_nodes))
    return 0
