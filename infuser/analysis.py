import ast
from itertools import combinations
import logging
import symtable
import sys
from typing import TypeVar

from infuser.cli import CLIPrinter
from .rules import WalkRulesVisitor, STAGES
from .unification import unify

T = TypeVar("T")
V = TypeVar("V")

logger = logging.getLogger(__name__)


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
        # for key in subs.keys(): 
        #     val = subs[key]
        #     import pdb; pdb.set_trace()    
        new_envs.append(type_env.substitute_types(subs))
        src_maps.append(srcs)

    for (e1, m1, s1), (e2, m2, s2) in \
            combinations(zip(new_envs, src_maps, STAGES), 2):
        for name in set(e1.keys()) & set(e2.keys()):
            import pdb; pdb.set_trace()
            if e1[name] != e2[name]:
                # TODO: Factor out the warnings I/O for easier I/O testing
                ast1 = m1.get(e1[name], set()) | m1.get(e2[name], set())
                ast2 = m2.get(e1[name], set()) | m2.get(e2[name], set())
                text = f"{s1} and {s2} disagree about {name.display_str()}"
                loc_set = set()
                loc_set.update((x.lineno - 1, x.col_offset) for x in ast1)
                loc_set.update((x.lineno - 1, x.col_offset) for x in ast2)
                printer.warn(text, sorted(loc_set))
    return 0
