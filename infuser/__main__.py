import argparse
import logging
import sys

from infuser.cli import CLIPrinter
from .analysis import analysis_main

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=bool)
    parser.add_argument("client_program_path", metavar="PATH",
                        type=argparse.FileType("r", encoding="utf-8"))
    parsed = parser.parse_args()

    code = parsed.client_program_path.read()
    printer = CLIPrinter(code, print_json=parsed.json)
    try:
        return analysis_main(code, parsed.client_program_path.name, printer)
    finally:
        parsed.client_program_path.close()


sys.exit(main())
