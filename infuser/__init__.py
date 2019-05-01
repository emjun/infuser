import ast
from typing import IO

from rules import walk_rules


def analysis_main(client: IO[str]):
    client.seek(0)
    client_ast = ast.parse(client.read(), client.name)

    # Just for demonstration. Remove the line below
    print(ast.dump(client_ast, annotate_fields=False))

    for match in walk_rules(client_ast):
        pass
