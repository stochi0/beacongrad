class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._op = _op
        self._children = set(_children)

    def __repr__(self):
        return f"Value(data={self.data}, _children={self._children}, _op={self._op})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

    def __truediv__(self, other):
        return Value(self.data / other.data, (self, other), '/')

    def __pow__(self, other):
        return Value(self.data ** other.data, (self, other), '**')

    def __sub__(self, other):
        return Value(self.data - other.data, (self, other), '-')