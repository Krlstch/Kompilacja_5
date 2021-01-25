from collections import defaultdict
import ast2
from SymbolTable import SymbolTable, VectorType

castable_types = ["int", "float"]


class NodeVisitor(object):
    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):  # Called if no explicit visitor function exists for a node.
        if isinstance(node, list):
            for elem in node:
                self.visit(elem)
        else:
            for child in node.children:
                if isinstance(child, list):
                    for item in child:
                        if isinstance(item, ast2.Ast):
                            self.visit(item)
                elif isinstance(child, ast2.Ast):
                    self.visit(child)

    # simpler version of generic_visit, not so general
    # def generic_visit(self, node):
    #    for child in node.children:
    #        self.visit(child)


class TypeChecker(NodeVisitor):
    def __init__(self):
        self.symbol_table = SymbolTable(None, 'program')
        self.depth = 0

    def visit_Program(self, node):
        for inst in node.instr:
            self.visit(inst)

    def visit_Print(self, node):
        self.visit(node.arg[0])

    def visit_Assign(self, node):
        type2 = self.visit(node.right)
        if type2 is None:
            return None
        if node.op == '=':
            self.symbol_table.put(node.left.name, type2)
        else:
            type1 = self.visit(node.left)
            if type1 is None:
                return None
            if type1 != type2:
                if type1 == "int" and type2 == "float":
                    self.symbol_table.put(node.left.name, type2)
                else:
                    print("Error at line {0}: Variables of incompatible types".format(node.line))
                    return None

    def visit_Arrassign(self, node):
        if len(node.arr) != 2:
            print("Error at line {0}: Access array must be of length 2".format(node.line))
            return None
        arr_type0 = self.visit(node.arr[0])
        arr_type1 = self.visit(node.arr[1])

        if arr_type0 != "int" or arr_type1 != "int":
            print("Error at line {0}: Access array elements must be integers".format(node.line))
            return None

        type1 = self.visit(node.left)
        if not isinstance(type1, VectorType):
            print("Error at line {0}: Can only access element of matrix".format(node.line))
            return None

        m_type = type1.type
        type2 = self.visit(node.right)
        if type2 != m_type:
            print("Error at line {0}: Variables of incompatible types".format(node.line))

    def visit_Access(self, node):
        if len(node.arr) != 2:
            print("Error at line {0}: Access array must be of length 2".format(node.line))
            return None
        arr_type0 = self.visit(node.arr[0])
        arr_type1 = self.visit(node.arr[1])

        if arr_type0 != "int" or arr_type1 != "int":
            print("Error at line {0}: Access array elements must be integers".format(node.line))
            return None
        type1 = self.visit(node.id)
        if not isinstance(type1, VectorType):
            print("Error at line {0}: Can only access element of matrix".format(node.line))
            return None
        return type1.type

    def visit_Binop(self, node):
        type1 = self.visit(node.left)
        type2 = self.visit(node.right)
        if type1 is None or type2 is None:
            return None

        if isinstance(type1, VectorType):
            if not isinstance(type2, VectorType):
                print("Error at line {0}: Variables of incompatible types".format(node.line))
                return None
            # Both matrix:
            if node.op in ["+", "-"]:
                if type1.width != type2.width or type1.height != type1.width:
                    print("Error at line {0}: Matrices of incompatible sizes".format(node.line))
                    return None
                if type1.type != type2.type:
                    print("Error at line {0}: Matrices of incompatible types".format(node.line))
                    return None
                return VectorType(width=type1.width, height=type1.height, type=type1.type)
            if node.op == "*":
                if type1.width != type2.height:
                    print("Error at line {0}: Matrices of incompatible sizes".format(node.line))
                if type1.type != type2.type:
                    print("Error at line {0}: Matrices of incompatible types".format(node.line))
                    return None
                return VectorType(width=type2.width, height=type1.height, type=type1.type)
            if node.op == "/":
                print("Error at line {0}: Matrices can not be divided")
                return None
        if isinstance(type2, VectorType):
            print("Error at line {0}: Variables of incompatible types".format(node.line))
            return None
        if type1 == "float" or type2 == "float":
            return "float"
        return "int"

    def visit_BinopMat(self, node):
        type1 = self.visit(node.left)
        type2 = self.visit(node.right)
        if type1 is None or type2 is None:
            return None

        if not isinstance(type1, VectorType) or not isinstance(type2, VectorType):
            print("Error at line {0}: Binary operation can only be done for matrices".format(node.line))
            return None

        if type1.width != type2.width or type1.height != type1.width:
            print("Error at line {0}: Matrices of incompatible sizes".format(node.line))
            return None
        if type1.type != type2.type:
            print("Error at line {0}: Matrices of incompatible types".format(node.line))
            return None

        return VectorType(width=type1.width, height=type1.height, type=type1.type)


    def visit_Relation(self, node):
        type1 = self.visit(node.left)
        type2 = self.visit(node.right)
        if type1 is None or type2 is None:
            return None

        if type1 != type2 and not (type1 in castable_types and type2 in castable_types):
            print("Error at line {0}: Variables of incompatible types".format(node.line))
            return None
        if isinstance(type1, VectorType) and node.op not in ["==", "!="]:
            print("Error at line {0}: Can not compare two matrices".format(node.line))
            return None
        return "bool"

    def visit_IfStatement(self, node):
        type1 = self.visit(node.cond)
        if type1 == "bool":
            self.symbol_table.push_scope("ifscope")
            self.visit(node.instr)
            self.symbol_table.pop_scope()
        else:
            print("Error at line {0}: Condition in if statement must be boolean".format(node.line))

    def visit_IfElseStatement(self, node):
        type1 = self.visit(node.cond)
        if type1 == "bool":
            self.symbol_table.push_scope("ifscope")
            self.visit(node.instr)
            self.symbol_table.pop_scope()
            self.symbol_table.push_scope("elsescope")
            self.visit(node.instr)
            self.symbol_table.pop_scope()
        else:
            print("Error at line {0}: Condition in if statement must be boolean".format(node.line))

    def visit_WhileLoop(self, node):
        type1 = self.visit(node.cond)
        if type1 == "bool":
            self.symbol_table.push_scope("loop")
            self.visit(node.instr)
            self.symbol_table.pop_scope()
        else:
            print("Error at line {0}: Condition in while loop must be boolean".format(node.line))

    def visit_ForLoop(self, node):
        type1 = self.visit(node.expr)
        type2 = self.visit(node.limit)
        if type1 == "int" and type2 == "int":
            self.symbol_table.push_scope("loop")
            self.symbol_table.put(node.id.name, type1)
            self.visit(node.instr)
            self.symbol_table.pop_scope()
        else:
            print("Error at line {0}: For loop's range must be integers ".format(node.line))

    def visit_Matrix(self, node):
        rows_count = len(node.mat)
        row_len = len(node.mat[0])
        m_type = self.visit(node.mat[0][0])

        for row in node.mat:
            if len(row) != row_len:
                print("Error at line {0}: Matrix rows must have the same length ".format(node.line))
                return None
            for element in row:
                if self.visit(element) != m_type:
                    print("Error at line {0}: Matrix elements must be of the same type".format(node.line))
                    return None
        return VectorType(width=row_len, height=rows_count, type=m_type)

    def visit_Scope(self, node):
        self.symbol_table.push_scope("scope")
        for inst in node.instr:
            self.visit(inst)
        self.symbol_table.pop_scope()

    def visit_BreakStatement(self, node):
        loop_scope = self.symbol_table.get_scope("loop")
        if loop_scope is None:
            print("Error at line {0}: Break ouside of loop scope".format(node.line))

    def visit_ContinueStatement(self, node):
        loop_scope = self.symbol_table.get_scope("loop")
        if loop_scope is None:
            print("Error at line {0}: Continue ouside of loop scope".format(node.line))

    def visit_ReturnStatement(self, node):
        self.visit(node.value)

    def visit_Uminus(self, node):
        return self.visit(node.expr)

    def visit_Variable(self, node):
        var = self.symbol_table.get(node.name)
        if var is not None:
            return var.type
        else:
            print("Error at line {0}: Variable {1} undeclared".format(node.line, node.name))
            return None

    def visit_Transposition(self, node):
        type1 = self.visit(node.mat)
        if isinstance(type1, VectorType):
            return VectorType(width=type1.height, height=type1.width, type=type1)
        else:
            print("Error at line {0}: Can only transpose matrix".format(node.line))
            return None

    def visit_Gen(self, node):
        if node.func == "eye":
            type1 = self.visit(node.arg)
            if type1 == "int":
                size = node.arg.value
                return VectorType(width=size, height=size, type="int")
            else:
                print("Error at line {0}: Function {1} argument must be int".format(node.line, node.func))
                return None
        else:
            if isinstance(node.arg, list):
                if len(node.arg) == 2:
                    type2 = self.visit(node.arg[0])
                    type3 = self.visit(node.arg[1])
                    if type2 == "int" and type3 == "int":
                        return VectorType(width=node.arg[0].value, height=node.arg[1].value, type="int")
                    else:
                        print("Error at line {0}: Function {1} arguments must be int".format(node.line, node.func))
                        return None
                else:
                    print("Error at line {0}: Function {1} must have 2 arguments".format(node.line, node.func))
                    return None
            else:
                print("Error at line {0}: Function {1} must have 2 arguments".format(node.line, node.func))
                return None

    def visit_IntNum(self, node):
        return "int"

    def visit_FloatNum(self, node):
        return "float"

    def visit_String(self, node):
        return "str"