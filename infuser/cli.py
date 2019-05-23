import json
import sys


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

    def warn(self, line: int, char_offset: int, text: str) -> None:
        if self.print_json:
            pkg = {"line": line, "offset": char_offset, "warning": text}
            json.dump(pkg, sys.stdout)
            sys.stdout.write("\n")
        else:
            if self.warnings_printed > 0:
                print()
            line = self.src_code.splitlines()[line]
            print(line)
            print((" " * char_offset) + "^")
            print((" " * min(2, char_offset)) + text)
        self.warnings_printed += 1

    def summarize(self):
        if self.warnings_printed > 0:
            print()
        print(f"Found {self.warnings_printed} potential problems")
