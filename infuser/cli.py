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

    def warn(self, text: str, locations: Sequence[Tuple[int, int]]) -> None:
        if self.print_json:
            pkg = {"warning": text,
                   "locations": [{"line": l[0], "offset": l[1]}
                                 for l in locations]}
            json.dump(pkg, sys.stdout)
            sys.stdout.write("\n")
        else:
            if self.warnings_printed > 0:
                print()
            for line_no, char_offset in locations:
                line = self.src_code.splitlines()[line_no]
                print(line)
                print((" " * char_offset) + "^")
            print((" " * min(2, char_offset)) + text)
        self.warnings_printed += 1

    def summarize(self):
        if self.warnings_printed > 0:
            print()
        print(f"Found {self.warnings_printed} potential problems")
