import ply.lex as lex

tokens = [
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'DOTPLUS',
    'DOTMINUS',
    'DOTTIMES',
    'DOTDIVIDE',
    'ASSIGN',
    'PLUSASSIGN',
    'MINUSASSIGN',
    'TIMESASSIGN',
    'DIVIDEASSIGN',
    'LESSER',
    'GREATER',
    'LESSEREQUAL',
    'GREATEREQUAL',
    'NOTEQUAL',
    'EQUAL',
    'LPAREN',
    'RPAREN',
    'LCURLY',
    'RCURLY',
    'LBRACET',
    'RBRACET',
    'RANGE',
    'TRANS',
    'COMMA',
    'SEMICOLON',
    "ID",
    "FLOAT",
    "INTEGER",
    "STRING",
    "COMMENT",

    "IFX",
    "IF",
    "ELSE",
    "FOR",
    "WHILE",
    "BREAK",
    "CONTINUE",
    "RETURN",
    "EYE",
    "ZEROS",
    "ONES",
    "PRINT"
]


reserved = {
    "if": "IF",
    "else": "ELSE",
    "for":  "FOR",
    "while": "WHILE",
    "break": "BREAK",
    "continue": "CONTINUE",
    "return": "RETURN",
    "eye": "EYE",
    "zeros": "ZEROS",
    "ones": "ONES",
    "print": "PRINT"
}

t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_DOTPLUS = r'\.\+'
t_DOTMINUS = r'\.-'
t_DOTTIMES = r'\.\*'
t_DOTDIVIDE = r'\./'
t_ASSIGN = r'='
t_PLUSASSIGN = r'\+='
t_MINUSASSIGN = r'-='
t_TIMESASSIGN = r'\*='
t_DIVIDEASSIGN = r'/='
t_LESSER = r'<'
t_GREATER = r'>'
t_LESSEREQUAL = r'<='
t_GREATEREQUAL = r'>='
t_NOTEQUAL = r'!='
t_EQUAL = r'=='
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LCURLY = r'\{'
t_RCURLY = r'\}'
t_LBRACET = r'\['
t_RBRACET = r'\]'
t_RANGE = r'\:'
t_TRANS = r'\''
t_COMMA = r'\,'
t_SEMICOLON = r'\;'


def t_COMMENT(t):
    r'\#.*\n'
    t.lexer.lineno += 1
    pass


def t_FLOAT(t):
    '(((([1-9][0-9]*)|0)[.][0-9]*)|([.][0-9]+))([eE][-+]?[0-9]*)?'
    t.value = float(t.value)
    return t


def t_INTEGER(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_STRING(t):
    r'\"[^\"]*\"'
    t.value = t.value[1:len(t.value) - 1]
    return t


def t_ID(t):
    r'[a-zA-Z_]\w*'
    t.type = reserved.get(t.value, 'ID')
    return t


def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


t_ignore = ' \t'


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


lexer = lex.lex()


def find_tok_column(input, token):
    line_start = input.rfind('\n', 0, token.lexpos) + 1
    return (token.lexpos - line_start) + 1