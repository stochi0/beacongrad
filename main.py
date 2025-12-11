from src.nn import Value
from src.viz import draw_dot

def main():
    print("Hello from autodiff!")
    a = Value(2.0, label='a')
    b = Value(3.0, label='b')
    c = a + b
    d = a * b
    e = c * d
    f = e / b
    f += a
    f += Value(1.0, label='1.0')
    print("f =", f)
    dot = draw_dot(f)
    dot.render('graph_before', format='svg', cleanup=True)
    f.backward()
    print("a.grad =", a.grad)
    print("b.grad =", b.grad)
    print("c.grad =", c.grad)
    print("d.grad =", d.grad)
    print("e.grad =", e.grad)
    print("f.grad =", f.grad)
    dot = draw_dot(f)
    dot.render('graph_after', format='svg', cleanup=True)

if __name__ == "__main__":
    main()
