import ast2
from Memory import *
from Exceptions import *
from visit import *
import numpy as np


def multiplication(a, b):
    if isinstance(a, np.array):
        return a @ b
    else:
        return a * b


class Interpreter(object):
    def __init__(self):
        self.scopes = MemoryStack()

    def interpret(self, node):
        self.visit(node)

    @on('node')
    def visit(self, node):
        pass

    @when(ast2.Program)
    def visit(self, node):
        self.scopes.push(Memory("Global"))
        for inst in node.instr:
            self.visit(inst)

    @when(ast2.Print)
    def visit(self, node):
        print(' '.join(str(self.visit(x)) for x in node.arg))

    @when(ast2.Assign)
    def visit(self, node):
        if node.op == "=":
            self.scopes.set(node.left.name, self.visit(node.right))
        else:
            new_value = {"+=": lambda a, b: a + b,
                         "-=": lambda a, b: a - b,
                         "*=": lambda a, b: a @ b if isinstance(a, np.array) else a * b,
                         "/=": lambda a, b: a / b}[node.op](self.scopes.get(node.left.name), self.visit(node.right))
            self.scopes.set(node.left.name, new_value)

    @when(ast2.Arrassign)
    def visit(self, node):
        mat = self.scopes.get(node.left.name)
        i = self.visit(node.arr[0])
        j = self.visit(node.arr[0])
        val = self.visit(node.right)
        new_value = {"+=": lambda a, b: a + b,
                     "-=": lambda a, b: a - b,
                     "*=": lambda a, b: a * b,
                     "/=": lambda a, b: a / b}[node.op](mat[i][j], val)
        mat[i][j] = new_value

    @when(ast2.Access)
    def visit(self, node):
        return self.scopes.get(node.left.name)[self.visit(node.arr[0])][self.visit(node.arr[1])]

    @when(ast2.Binop)
    def visit(self, node):
        return {"+": lambda a, b: a + b,
                "-": lambda a, b: a - b,
                "*": lambda a, b: a @ b if isinstance(a, np.array) else a * b,
                "/": lambda a, b: a / b}[node.op](self.visit(node.left), self.visit(node.right))

    @when(ast2.BinopMat)
    def visit(self, node):
        return {".+": lambda a, b: a + b,
                ".-": lambda a, b: a - b,
                ".*": lambda a, b: a * b,
                "./": lambda a, b: a / b}[node.op](self.visit(node.left), self.visit(node.right))

    @when(ast2.Relation)
    def visit(self, node):
        return {">": lambda a, b: a > b,
                ">=": lambda a, b: a >= b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                "<=": lambda a, b: a <= b,
                "<": lambda a, b: a < b}[node.op](self.visit(node.left), self.visit(node.right))

    @when(ast2.IfStatement)
    def visit(self, node):
        if self.visit(node.cond):
            self.scopes.push(Memory("If"))
            try:
                self.visit(node.instr)
            except BreakException:
                self.scopes.pop()
                raise BreakException()
            except ContinueException:
                self.scopes.pop()
                raise ContinueException()
            self.scopes.pop()

    @when(ast2.IfElseStatement)
    def visit(self, node):
        if self.visit(node.cond):
            self.scopes.push(Memory("If"))
            try:
                self.visit(node.instr)
            except BreakException:
                self.scopes.pop()
                raise BreakException()
            except ContinueException:
                self.scopes.pop()
                raise ContinueException()
            self.scopes.pop()
        else:
            self.scopes.push(Memory("Else"))
            try:
                self.visit(node.else_instr)
            except BreakException:
                self.scopes.pop()
                raise BreakException()
            except ContinueException:
                self.scopes.pop()
                raise ContinueException()
            self.scopes.pop()

    @when(ast2.WhileLoop)
    def visit(self, node):
        self.scopes.push(Memory("WhileLoop"))
        cond = self.visit(node.cond)
        while cond:
            try:
                self.visit(node.instr)
            except BreakException:
                break
            except ContinueException:
                continue
            cond = self.visit(node.cond)
        self.scopes.pop()

    @when(ast2.ForLoop)
    def visit(self, node):
        self.scopes.push(Memory("ForLoop"))
        self.scopes.insert(node.id.name, self.visit(node.expr))
        limit = self.visit(node.limit)
        while self.scopes.get(node.id.name) != limit:
            try:
                self.visit(node.instr)
            except BreakException:
                break
            except ContinueException:
                continue
            self.scopes.set(node.id.name, self.scopes.get(node.id.name) + 1)
        self.scopes.pop()

    @when(ast2.Variable)
    def visit(self, node):
        return self.scopes.get(node.name)

    @when(ast2.Matrix)
    def visit(self, node):
        return np.arry(self.visit(node.mat))

    @when(ast2.Scope)
    def visit(self, node):
        for inst in node.instr:
            self.visit(inst)

    @when(ast2.BreakStatement)
    def visit(self, node):
        raise BreakException()

    @when(ast2.ContinueStatement)
    def visit(self, node):
        raise ContinueException()

    @when(ast2.ReturnStatement)
    def visit(self, node):
        raise ReturnValueException(self.visit(node.value))

    @when(ast2.Uminus)
    def visit(self, node):
        return -self.visit(node.expr)

    @when(ast2.Transposition)
    def visit(self, node):
        return np.transpose(self.visit(node.mat))

    @when(ast2.Gen)
    def visit(self, node):
        if node.func == "eye":
            value = self.visit(node.value)
            return np.eye(shape=value)
        else:
            value1 = self.visit(node.value[0])
            value2 = self.visit(node.value[1])
            return {"ones": lambda a, b: np.ones(shape=[a, b]),
                    "zeros": lambda a, b: np.zeros(shape=[a, b])}[node.func](value1, value2)

    @when(ast2.IntNum)
    def visit(self, node):
        return node.value

    @when(ast2.FloatNum)
    def visit(self, node):
        return node.value

    @when(ast2.String)
    def visit(self, node):
        return node.string
