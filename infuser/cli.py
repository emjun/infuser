from itertools import groupby
import json
import sys
from typing import Tuple, Sequence


class CLIPrinter:
    """Produces Infuser warnings output.

    >>> src = '''import sys
    >>> sys.good_thing()
    >>> if 1:
    >>>     sys.bad_thing()'''
    >>> printer = CLIPrinter(src)
    >>> printer.warn(0, 0, "Oh no! An import!")
    >>> printer.warn(3, 4, "Oh no! A bad thing!")
    >>> printer.summarize()
    """

    def __init__(self, src_code: str, print_json: bool = False):
        super().__init__()
        self.src_code = src_code
        self.print_json = print_json
        self.warnings_printed = 0

    def warn(self, n1, n2, locations: Sequence[Tuple[int, int]]) -> None:
        text = f"Disagreement about {n1.display_str()} and {n2.display_str()}"
        if self.print_json:
            pkg = {"warning": text,
                   "locations": [{"line": l[0], "offset": l[1]}
                                 for l in locations]}
            json.dump(pkg, sys.stdout)
            sys.stdout.write("\n")
        else:
            if self.warnings_printed > 0:
                print()
            print(f"{text}. Interactions found at:")
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
        if self.warnings_printed > 0:
            print()
        print(f"Found {self.warnings_printed} potential problems")
