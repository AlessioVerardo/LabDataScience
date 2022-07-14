import sys
from typing import Iterator

def fibo(n: int) -> Iterator[int]:
    """Yields a generator of the fibonacci
    sequence, up to the nth value"""
    a, b = 0, 1
    for i in range(n):
        yield a
        b, a = a, a+b

if __name__ == '__main__':
    try:
        n = int(sys.argv[1])
    except ValueError:
        raise ValueError("Wrong input, provide an int")
    print(f"{[x for x in fibo(n)]}")
