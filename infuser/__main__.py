import argparse
import logging

from . import analysis_main

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


main()
