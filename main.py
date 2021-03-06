import sys
import ply.yacc as yacc
import Mparser2
import scanner
from TypeChecker import TypeChecker
from Interpreter import Interpreter

if __name__ == '__main__':

    try:
        filename = sys.argv[1] if len(sys.argv) > 1 else "sqrt.m"
        file = open(filename, "r")
    except IOError:
        print("Cannot open {0} file".format(filename))
        sys.exit(0)

    parser = Mparser2.parser
    text = file.read()
    ast = parser.parse(text, lexer=scanner.lexer)

    # Below code shows how to use visitor
    typeChecker = TypeChecker()
    typeChecker.visit(ast)  # or alternatively ast.accept(typeChecker)

    interpreter = Interpreter()
    interpreter.interpret(ast)