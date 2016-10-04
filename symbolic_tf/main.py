from __future__ import print_function
from __future__ import division

from graph import Graph
from session import Session
from ops import Add, Sub, Mul, Div, Dot, Variable, Placeholder, Constant

def main():
    graph = Graph()
    sess = Session(graph)

    a = Variable(4)
    b = Constant(3)
    c = Placeholder()

    d = Add(Mul(a, b), c)

    result = sess.run(d, feed_dict={
        c: 3
    })
    print(result) # 15

if __name__ == '__main__':
    main()
