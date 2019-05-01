import ast
import argparse
import logging
from typing import TextIO

from rules import walk_rules

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("client_program_path", metavar="PATH",
                        type=argparse.FileType("r", encoding="utf-8"))
    parsed = parser.parse_args()

    try:
        analysis_main(parsed.client_program_path)
    finally:
        parsed.client_program_path.close()


def analysis_main(client: TextIO):
    client.seek(0)
    client_ast = ast.parse(client.read(), client.name)
    for match in walk_rules(client_ast):
        pass

    # Just for demonstration. Remove the line below
    print(ast.dump(client_ast, annotate_fields=False))


main()
