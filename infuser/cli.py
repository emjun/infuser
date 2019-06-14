from itertools import groupby
import json
import sys
from typing import Tuple, Set


class CLIPrinter:
    """Produces Infuser warnings output.
    """

    def __init__(self, src_code: str, print_json: bool = False):
        super().__init__()
        self.src_code = src_code
        self.print_json = print_json
        self.warnings_printed = 0

    def warn(self, n1, n2, locations: Set[Tuple[int, int]],
             problem_stage: str) -> None:

        locations = sorted(locations)

        text = f"Types of {n1.display_str()} and {n2.display_str()} were " \
               f"unified in {problem_stage} only"
        if self.print_json:
            pkg = {"warning": text,
                   "locations": [{"line": l[0], "offset": l[1]}
                                 for l in locations]}
            json.dump(pkg, sys.stdout)
            sys.stdout.write("\n")
        else:
            if self.warnings_printed > 0:
                print()
            print(f"{text}. See:")
            print()

            for line_no, items in groupby(sorted(locations), key=lambda t: t[0]):
                char_offsets = [o for _, o in items]
                line = self.src_code.splitlines()[line_no]
                line_prefix = f"  line {line_no}: "
                print(line_prefix + line.lstrip())
                chars_removed = len(line) - len(line.lstrip())
                ptr_line = " " * (len(line_prefix) + char_offsets[0] - chars_removed)
                ptr_line += "^"
                for x, y in zip(char_offsets, char_offsets[1:]):
                    ptr_line += " " * ((y - x) - 1) + "^"
                print(ptr_line)
        self.warnings_printed += 1

    def summarize(self):
        # if self.warnings_printed > 0:
            # print()
        print(f"Found {self.warnings_printed} potential problems")
