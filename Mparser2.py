import numpy as np

import scanner
import ply.yacc as yacc

from ast2 import *

tokens = scanner.tokens

precedence = (
    ('left', 'ASSIGN'),
    ('nonassoc', 'LESSER', 'GREATER', 'LESSEREQUAL', 'GREATEREQUAL'),
    ("left", 'PLUS', 'MINUS'),
    ("left", "TIMES", "DIVIDE"),
    ("left", 'DOTPLUS', 'DOTMINUS'),
    ("left", "DOTTIMES", "DOTDIVIDE"),
    ('right', 'UMINUS'),
    ("left", "TRANS"),
    ("nonassoc", "IFX"),
    ("nonassoc", "ELSE")
)


def p_error(p):
    if p:
        print("Syntax error at line {0}: LexToken({1}, '{2}')".format(p.lineno, p.type, p.value))
    else:
        print("Unexpected end of input")


def p_program(t):
    """program : instructions"""
    t[0] = Program(instr=t[1])


def p_instructions(t):
    """instructions : instructions instruction
                | """
    if len(t) > 1:
        t[1].append(t[2])
        t[0] = t[1]
    else:
        t[0] = []


def p_empty_instruction(t):
    """instruction : SEMICOLON"""
    pass


def p_expression_value1(t):
    """expression : INTEGER"""
    t[0] = IntNum(value=t[1], line=t.lineno(1))


def p_expression_value2(t):
    """expression : FLOAT"""
    t[0] = FloatNum(value=t[1], line=t.lineno(1))


def p_expression_value3(t):
    """expression : matrix"""
    t[0] = t[1]


def p_expression_value4(t):
    """expression : STRING"""
    t[0] = String(string=t[1], line=t.lineno(1))


def p_expression_ID(t):
    """expression : ID"""
    t[0] = Variable(name=t[1], line=t.lineno(1))


def p_group_expression(t):
    """expression : LPAREN expression RPAREN"""
    t[0] = t[2]


def p_instructions_scope(t):
    """instruction : LCURLY instructions RCURLY"""
    t[0] = Scope(instr=t[2], line=t.lineno(1))


def p_expression_binop(t):
    """expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression"""
    t[0] = Binop(left=t[1], op=t[2], right=t[3], line=t.lineno(2))


def p_expression_binop_mat(t):
    """expression : expression DOTPLUS expression
                  | expression DOTMINUS expression
                  | expression DOTTIMES expression
                  | expression DOTDIVIDE expression"""
    t[0] = BinopMat(left=t[1], op=t[2], right=t[3], line=t.lineno(2))


def p_expression_relation(t):
    """condition : expression LESSER expression
                | expression GREATER expression
                | expression LESSEREQUAL expression
                | expression GREATEREQUAL expression
                | expression NOTEQUAL expression
                | expression EQUAL expression"""
    t[0] = Relation(left=t[1], op=t[2], right=t[3], line=t.lineno(1))


def p_uminus(t):
    """expression : MINUS expression %prec UMINUS"""
    t[0] = Uminus(expr=t[2], line=t.lineno(1))


def p_trans(t):
    """matrix : expression TRANS"""
    t[0] = Transposition(mat=t[1], line=t.lineno(1))


def p_matrix_gen(t):
    """matrix : ZEROS LPAREN expression RPAREN
              | ONES LPAREN list RPAREN
              | EYE LPAREN list RPAREN"""
    t[0] = Gen(func=t[1], arg=t[3], line=t.lineno(1))


def p_assign(t):
    """instruction : ID ASSIGN expression SEMICOLON
                    | ID PLUSASSIGN expression SEMICOLON
                    | ID MINUSASSIGN expression SEMICOLON
                    | ID TIMESASSIGN expression SEMICOLON
                    | ID DIVIDEASSIGN expression SEMICOLON"""
    t[0] = Assign(left=Variable(t[1], line=t.lineno(1)), op=t[2], right=t[3], line=t.lineno(1))


def p_position_assign(t):
    """instruction : ID array ASSIGN expression SEMICOLON
                   | ID array PLUSASSIGN expression SEMICOLON
                   | ID array MINUSASSIGN expression SEMICOLON
                   | ID array TIMESASSIGN expression SEMICOLON
                   | ID array DIVIDEASSIGN expression SEMICOLON"""  # A[0,1] = 5, etc.
    t[0] = Arrassign(left=Variable(t[1], line=t.lineno(1)), arr=t[2], op=t[3], right=t[4], line=t.lineno(1))


def p_if_else(t):
    """instruction : IF LPAREN condition RPAREN instruction %prec IFX
                  | IF LPAREN condition RPAREN instruction ELSE instruction"""
    if len(t) == 6:
        t[0] = IfStatement(cond=t[3], instr=t[5], line=t.lineno(1))
    else:
        t[0] = IfElseStatement(cond=t[3], instr=t[5], else_instr=t[7], line=t.lineno(1))


def p_while(t):
    """instruction : WHILE LPAREN condition RPAREN instruction"""
    t[0] = WhileLoop(cond=t[3], instr=t[5], line=t.lineno(1))


def p_for(t):
    """instruction : FOR ID ASSIGN expression RANGE expression instruction"""
    t[0] = ForLoop(id=Variable(t[2], t.lineno(1)), expr=t[4], limit=t[6], instr=t[7], line=t.lineno(1))


def p_special_instruction(t):
    """instruction : BREAK SEMICOLON
                   | CONTINUE SEMICOLON
                   | RETURN expression SEMICOLON"""
    if t[1] == "break":
        t[0] = BreakStatement(line=t.lineno(1))
    elif t[1] == "continue":
        t[0] = ContinueStatement(line=t.lineno(1))
    elif t[1] == "return":
        t[0] = ReturnStatement(value=t[2], line=t.lineno(1))


def p_print(t):
    """instruction : PRINT list SEMICOLON"""
    t[0] = Print(arg=t[2], line=t.lineno(1))


def p_matrix(t):
    """matrix : LBRACET arraylist RBRACET"""
    t[0] = Matrix(mat=t[2], line=t.lineno(1))


def p_arraylist(t):
    """arraylist : array
                 | arraylist COMMA array"""
    if len(t) > 2:
        t[1].append(t[3])
        t[0] = t[1]
    else:
        t[0] = [t[1]]


def p_array(t):
    """array : LBRACET list RBRACET"""
    t[0] = t[2]


def p_list(t):
    """list : expression
            | list COMMA expression"""
    if len(t) > 2:
        t[1].append(t[3])
        t[0] = t[1]
    else:
        t[0] = [t[1]]


def p_array_access(t):
    """expression : ID array"""  # A[0,1], B[1], etc.
    t[0] = Access(id=Variable(t[1], line=t.lineno(1)), arr=t[2], line=t.lineno(1))


parser = yacc.yacc()