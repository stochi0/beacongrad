from src.nn import Value

def main():
    print("Hello from autodiff!")
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    print(c)


if __name__ == "__main__":
    main()
