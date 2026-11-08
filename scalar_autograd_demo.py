#### Educational: scalar autograd from scratch, checked against finite differences
"""
A minimal, scalar-valued reverse-mode autodiff engine.

Each Value stores its data, its gradient, the Values it was computed from,
and a closure (_backward) that pushes its gradient to those parents using
the local derivative (the chain rule). backward() topologically sorts the
graph and runs the closures from output to inputs.

Key ideas demonstrated:
  * gradients ACCUMULATE (+=) when a value is used more than once
  * backward order must be reverse topological
  * results match numerical (finite-difference) gradients

Run:  python scalar_autograd_demo.py
"""

import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    # Derived ops reuse the primitives above, so they get gradients for free.
    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def backward(self):
        order, seen = [], set()

        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def numerical_grad(f, xs, h=1e-6):
    """Central finite differences of scalar f w.r.t. each float in xs."""
    grads = []
    for i in range(len(xs)):
        up = list(xs); up[i] += h
        dn = list(xs); dn[i] -= h
        grads.append((f(*up) - f(*dn)) / (2 * h))
    return grads


def demo_reuse():
    print("1) Gradient accumulation: f(x) = x*x + x  ->  f'(x) = 2x + 1")
    x = Value(3.0)
    y = x * x + x
    y.backward()
    print(f"   x=3: autograd={x.grad}, expected={2 * 3 + 1}\n")


def demo_neuron():
    print("2) A single tanh neuron: out = tanh(w1*x1 + w2*x2 + b)")
    vals = dict(x1=2.0, x2=0.0, w1=-3.0, w2=1.0, b=6.88)
    v = {k: Value(d) for k, d in vals.items()}
    out = (v["w1"] * v["x1"] + v["w2"] * v["x2"] + v["b"]).tanh()
    out.backward()

    def f(x1, x2, w1, w2, b):
        return math.tanh(w1 * x1 + w2 * x2 + b)

    num = numerical_grad(f, list(vals.values()))
    for (name, val), n in zip(v.items(), num):
        print(f"   d out/d {name:2s}: autograd={val.grad:+.6f}  numeric={n:+.6f}")
    print()


def demo_regression(steps=200, lr=0.05):
    print("3) Fit y = 2x + 1 with gradient descent on MSE")
    data = [(x, 2 * x + 1) for x in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    w, b = Value(0.0), Value(0.0)
    for step in range(steps):
        loss = sum(((w * x + b) - y) ** 2 for x, y in data) / len(data)
        w.grad = b.grad = 0.0  # gradients accumulate, so reset each step
        loss.backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
        if step % 50 == 0 or step == steps - 1:
            print(f"   step {step:3d}: loss={loss.data:.6f}  w={w.data:.4f}  b={b.data:.4f}")


if __name__ == "__main__":
    demo_reuse()
    demo_neuron()
    demo_regression()
