import numpy as np
from dataclasses import dataclass

symbols = {}


@dataclass
class Ast(object):
    pass


class Program(Ast):
    def __init__(self, instr):
        self.instr = instr


class Print(Ast):
    def __init__(self, arg, line):
        self.arg = arg
        self.line = line


class Assign(Ast):
    def __init__(self, left, op, right, line):
        self.left = left
        self.op = op
        self.right = right
        self.line = line

    # def execute(self):
    #     if self.params[0] == "=":
    #         symbols[self.params[1]] = Ast.resolve(self.params[2])
    #     elif self.params[0] == "+=":
    #         symbols[self.params[1]] = symbols[self.params[1]] + Ast.resolve(self.params[2])
    #     elif self.params[0] == "-=":
    #         symbols[self.params[1]] = symbols[self.params[1]] - Ast.resolve(self.params[2])
    #     elif self.params[0] == "*=":
    #         symbols[self.params[1]] = symbols[self.params[1]] * Ast.resolve(self.params[2])
    #     elif self.params[0] == "/=":
    #         symbols[self.params[1]] = symbols[self.params[1]] / Ast.resolve(self.params[2])


class Arrassign(Ast):
    def __init__(self, left, arr, op, right, line):
        self.left = left
        self.op = op
        self.arr = arr
        self.right = right
        self.line = line

    # def execute(self):
    #     if self.params[0] == "=":
    #         symbols[self.params[1]][self.params[2][0]][self.params[2][1]] = Ast.resolve(self.params[3])
    #     elif self.params[0] == "+=":
    #         symbols[self.params[1]][self.params[2][0]][self.params[2][1]] += Ast.resolve(self.params[3])
    #     elif self.params[0] == "-=":
    #         symbols[self.params[1]][self.params[2][0]][self.params[2][1]] -= Ast.resolve(self.params[3])
    #     elif self.params[0] == "*=":
    #         symbols[self.params[1]][self.params[2][0]][self.params[2][1]] *= Ast.resolve(self.params[3])
    #     elif self.params[0] == "/=":
    #         symbols[self.params[1]][self.params[2][0]][self.params[2][1]] /= Ast.resolve(self.params[3])


#
class Access(Ast):
    def __init__(self, id, arr, line):
        self.id = id
        self.arr = arr
        self.line = line

    # def execute(self):
    #     result = symbols[self.params[0]][self.params[1][0]][self.params[1][1]]
    #     return result


class Binop(Ast):
    def __init__(self, left, op, right, line):
        self.left = left
        self.op = op
        self.right = right
        self.line = line


    # def execute(self):
    #     result = {
    #         '+': lambda a, b: a + b,
    #         '-': lambda a, b: a - b,
    #         '*': lambda a, b: a * b,
    #         '/': lambda a, b: a / b
    #     }[self.params[0]](Ast.resolve(self.params[1]), Ast.resolve(self.params[2]))
    #     return result


class BinopMat(Ast):
    def __init__(self, left, op, right, line):
        self.left = left
        self.op = op
        self.right = right
        self.line = line

    # def execute(self):
    #     result = {
    #         '>': lambda a, b: (a > b),
    #         '>=': lambda a, b: (a >= b),
    #         '<': lambda a, b: (a < b),
    #         '<=': lambda a, b: (a <= b),
    #         '==': lambda a, b: (a == b),
    #         '!=': lambda a, b: (a != b)
    #     }[self.params[0]](Ast.resolve(self.params[1]), Ast.resolve(self.params[2]))
    #     return result


class Relation(Ast):
    def __init__(self, left, op, right, line):
        self.left = left
        self.op = op
        self.right = right
        self.line = line

    # def execute(self):
    #     result = {
    #         '>': lambda a, b: (a > b),
    #         '>=': lambda a, b: (a >= b),
    #         '<': lambda a, b: (a < b),
    #         '<=': lambda a, b: (a <= b),
    #         '==': lambda a, b: (a == b),
    #         '!=': lambda a, b: (a != b)
    #     }[self.params[0]](Ast.resolve(self.params[1]), Ast.resolve(self.params[2]))
    #     return result


class IfStatement(Ast):
    def __init__(self, cond, instr, line):
        self.cond = cond
        self.instr = instr
        self.line = line

    # def execute(self):
    #     result = None
    #     if Ast.resolve(self.params[0]):
    #         result = Ast.resolve(self.params[1])
    #     return result


class IfElseStatement(Ast):

    def __init__(self, cond, instr, else_instr, line):
        self.cond = cond
        self.instr = instr
        self.else_instr = else_instr
        self.line = line
        
    # def execute(self):
    #     if Ast.resolve(self.params[0]):
    #         result = Ast.resolve(self.params[1])
    #     else:
    #         result = Ast.resolve(self.params[2])
    #     return result


class WhileLoop(Ast):
    def __init__(self, cond, instr, line):
        self.cond = cond
        self.instr = instr
        self.line = line

    # def execute(self):
    #     result = None
    #     while Ast.resolve(self.params[0]):
    #         print('|', end='  ')
    #         result = Ast.resolve(self.params[1])
    #         if result == "continue":
    #             continue
    #         if result == "break":
    #             break
    #     return result


class ForLoop(Ast):
    def __init__(self, id, expr, limit, instr, line):
        self.id = id
        self.expr = expr
        self.limit = limit
        self.instr = instr
        self.line = line

    # def execute(self):
    #     result = None
    #     symbols[self.params[0]] = Ast.resolve(self.params[1])
    #     while symbols[self.params[0]] <= Ast.resolve(self.params[2]):
    #         print('|', end='  ')
    #         result = Ast.resolve(self.params[3])
    #         if result == "continue":
    #             continue
    #         if result == "break":
    #             break
    #         symbols[self.params[0]] += 1
    #     return result


class Variable(Ast):
    def __init__(self, name, line):
        self.name = name
        self.line = line

    # def execute(self):
    #     result = symbols.get(self.params)
    #     return result


class Matrix(Ast):
    def __init__(self, mat, line):
        self.mat = mat
        self.line = line

    # def execute(self):
    #     pass


class Scope(Ast):
    def __init__(self, instr, line):
        self.instr = instr
        self.line = line


    # def execute(self):
    #     result = None
    #     for instr in self.params:
    #         result = Ast.resolve(instr)
    #         if result in ["continue", "break"]:
    #             break
    #     return result


class BreakStatement(Ast):
    def __init__(self, line):
        self.line = line


    # def execute(self):
    #     result = "break"
    #     return result


class ContinueStatement(Ast):
    def __init__(self, line):
        self.line = line

    # def execute(self):
    #     result = "continue"
    #     return result


class ReturnStatement(Ast):
    def __init__(self, value, line):
        self.value = value
        self.line = line

    # def execute(self):
    #     exit(self.params)


class Uminus(Ast):
    def __init__(self, expr, line):
        self.expr = expr
        self.line = line

    # def execute(self):
    #     result = - Ast.resolve(self.params)
    #     return result


class Transposition(Ast):
    def __init__(self, mat, line):
        self.mat = mat
        self.line = line

    # def execute(self):
    #     result = np.transpose(Ast.resolve(self.params))
    #     return result


class Gen(Ast):
    def __init__(self, func, arg, line):
        self.func = func
        self.arg = arg
        self.line = line


class IntNum(Ast):
    def __init__(self, value, line):
        self.value = value
        self.line = line


class FloatNum(Ast):
    def __init__(self, value, line):
        self.value = value
        self.line = line


class String(Ast):
    def __init__(self, string, line):
        self.string = string
        self.line = line
