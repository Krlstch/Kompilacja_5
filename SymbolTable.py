class VariableSymbol:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __str__(self):
        return "name: {0}\ttype: {1}".format(self.name, self.type)


class VectorType:
    def __init__(self, width, height, type):
        self.width = width
        self.height = height
        self.type = type

    def __str__(self):
        return 'matrix: size: {0},{1}\ttype: {2}'.format(self.width, self.height, self.type)


class SymbolTable:
    def __init__(self, parent, name):
        self.parent = parent
        self.last_scope = 0
        self.scopes = [({}, "global")]  # stack of scopes of variables (dicts)
        self.name = name

    def put(self, name, symbol):  # put variable symbol or fundef under <name> entry
        it = self.last_scope
        self.scopes[it][0][name] = VariableSymbol(name, symbol)

    def get(self, name):  # get variable symbol or fundef from <name> entry
        it = self.last_scope
        while it >= 0:
            if name in self.scopes[it][0]:
                return self.scopes[it][0][name]
            it -= 1
        return None

    def get_scope(self, name):
        it = self.last_scope
        while it >= 0:
            if self.scopes[it][1] == name:
                return self.scopes[it]
            it -= 1
        return None

    def get_parent_scope(self):
        return self.parent

    def push_scope(self, name):
        self.scopes.append(({}, name))
        self.last_scope += 1

    def pop_scope(self):
        self.scopes.pop()
        self.last_scope -= 1
