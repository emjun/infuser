import ast
import symtable
from typing import IO

from .rules import WalkRulesVisitor


def partition_str(code_str: str):
    wrangling_key = 'infuser.wrangling()'
    analysis_key = 'infuser.analyzing()'
    # Idea 1: could also just split the str rep of the script into 2 ASTs before walking down...
    preamble = code_str.partition(wrangling_key)[0]
    tmp_wrangling_str = code_str.partition(wrangling_key)[2]
    wrangling_str = tmp_wrangling_str.partition(analysis_key)[0]
    analysis_str = tmp_wrangling_str.partition(analysis_key)[2]

    return {    'preamble': preamble, 
                'wrangling': wrangling_str,
                'analysis': analysis_str}


    # Idea 2: could walk down and then cut AST

def analysis_main(client: IO[str]):
    client.seek(0)
    code_str = client.read()

    table = symtable.symtable(code_str, client.name, 'exec')

    # First, split script into wrangling and analysis
    stages_code_str = partition_str(code_str)
    # Preamble is all the code before wrangling
    preamble_str = stages_code_str['preamble']
    # Wrangling is all the code between 'infuser.wrangling()' and 'infuser.analyzing()' in the original script
    wrangling_str = stages_code_str['wrangling']
    # Analysis is all the code after 'infuser.analyzing()' in the original script
    analysis_str = stages_code_str['analysis']

    # client_ast = ast.parse(code_str, client.name)
    preamble_ast = ast.parse(preamble_str, client.name)
    wrangling_ast = ast.parse(wrangling_str, client.name)
    analysis_ast = ast.parse(analysis_str, client.name)

    # Second, visit wrangling and analysis separately
    visitor = WalkRulesVisitor(table)
    # visitor.visit(client_ast)
    # TODO Can we do this? Can Visitor be visited twice???
    visitor.visit(wrangling_ast)
    visitor.visit(analysis_ast)

    # Third, unify on each side

    # Fourth, compare sides
    # Fifth, surface to user....

    raise NotImplementedError("analysis_main is incomplete")
