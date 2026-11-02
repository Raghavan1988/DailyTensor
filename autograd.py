#### Educational: Reverse-mode automatic differentiation (autograd)
"""
You already wrote the backward pass of attention by hand in
backprop_attention.py. That works, but it does not scale: every new
layer needs a matching backward, and every new graph (skip connections,
weight sharing, dynamic control flow) needs you to apply the SUM rule
and the chain rule correctly.

Autograd is the compiler for that. During the forward pass it records a
graph of primitive ops. Each primitive knows only its own local
vector-Jacobian product (VJP). `.backward()` walks the graph in reverse
topological order, multiplying the incoming gradient by each local VJP.

Why reverse mode, not forward mode?
    ML has few outputs (usually a scalar loss) and millions of inputs
    (parameters). Reverse-mode autodiff computes dL/d(every parameter)
    in one backward pass. Forward-mode would need one pass PER parameter.

    The thing we actually compute is a VJP, not a Jacobian:

        if  y = f(x)  and we already have  g = dL/dy,
        the VJP is  g @ J_f  =  dL/dx

    We never materialize J_f. For softmax that would be an (n, n, n)
    tensor; the VJP is a few elementwise ops (see Softmax.backward).

The three rules this engine is made of (same ones as backprop_attention.py):

    Chain rule:   dL/dx = dL/dy * dy/dx     (implemented as a VJP, not a
                                            materialized Jacobian)
    Matmul rule:  Z = U @ W  =>  dU = dZ @ W^T ,  dW = U^T @ dZ
    Sum rule:     if x is used twice, dx = dx_path1 + dx_path2

This file builds the machine in two stages:

    Part 1 — Scalar `Value`. One number, one gradient. The whole idea
             without shapes, broadcasting, or matmuls.

    Part 2 — Tensor autograd. NumPy arrays + a PyTorch-style
             `Function.apply` protocol. Enough to train a tiny MLP and
             to reproduce the attention gradients you derived by hand.

PyTorch name map:

    this file                  torch
    ------------------------   ---------------------------------
    Tensor                     torch.Tensor
    Tensor.grad                tensor.grad
    Tensor.requires_grad       tensor.requires_grad
    Tensor._ctx / .grad_fn     tensor.grad_fn
    Function                   torch.autograd.Function
    Function.apply             Function.apply
    Function.save_for_backward ctx.save_for_backward
    no_grad                    torch.no_grad
"""

from __future__ import annotations

import numpy as np


# ======================================================================
# Part 1 — Scalar autograd
# ======================================================================
#
# A Value is a single float that remembers:
#   - which Values it was built from  (_parents)
#   - how to send its gradient back to those parents  (_backward)
#
# The graph is implicit: parent pointers. `.backward()` topologically
# sorts that graph, seeds the output gradient with 1, and walks reverse.


class Value:
    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(_parents)
        self._op = _op          # debug only: "+", "*", "relu", ...
        self._backward = lambda: None

    def __repr__(self):
        return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, so the whole upstream gradient flows to both.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a. Multiply by upstream (chain rule).
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __truediv__(self, other):
        return self * (other ** -1.0)

    def __pow__(self, p):
        p = float(p)
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            # d(x^p)/dx = p * x^(p-1)
            self.grad += p * (self.data ** (p - 1.0)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "relu")

        def _backward():
            # Subgradient at 0 is the interval [0, 1]; we pick 0, like PyTorch.
            self.grad += (out.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Reverse-mode autodiff. Must be called on a scalar (this class
        is all scalars). Seeds dL/dL = 1 and walks parents in reverse
        topological order so every node sees a fully-accumulated gradient
        before it fires its own _backward.
        """
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents:
                    build(p)
                topo.append(v)          # post-order: children before parent

        build(self)
        self.grad = 1.0                 # dL/dL = 1
        for v in reversed(topo):
            v._backward()


# ======================================================================
# Part 2 — Tensor autograd (PyTorch-shaped)
# ======================================================================
#
# Two objects, same split PyTorch uses:
#
#   Tensor   — the value. Holds data, grad, and a pointer to the Function
#              that created it (grad_fn). Leaves (parameters, inputs)
#              have grad_fn is None.
#
#   Function — the op. forward(inputs) -> output data, and
#              backward(dL/d output) -> VJPs for each input.
#              Function.apply is the only public entry: it runs forward,
#              wraps the result in a Tensor, and (if any input needs
#              grad) hangs this Function off the result as grad_fn.
#
# The engine (Tensor.backward) is deliberately op-agnostic: it only
# knows how to topo-sort, call Function.backward, unbroadcast, and
# accumulate. New ops = new Function subclasses. That is the whole
# extension model.


_GRAD_ENABLED = True


class no_grad:
    """Temporarily do not record ops. Same idea as torch.no_grad()."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev


def unbroadcast(grad, shape):
    """Reverse of NumPy broadcasting.

    Forward:  (3, 1) + (1, 4) -> (3, 4)
    Backward: the (3, 4) upstream gradient must be summed back down to
    each input's original shape — extra dims (ones that broadcasting
    introduced) are SUMMED, because each of those outputs depended on
    the same input element (sum rule).
    """
    grad = np.asarray(grad, dtype=np.float64)
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Function:
    def __init__(self):
        self.saved = ()
        self.parents = ()

    def save_for_backward(self, *arrays):
        # Copy so a later in-place update of a leaf (SGD: p.data -= lr*g)
        # cannot silently corrupt the activations this backward needs.
        # Real PyTorch tracks version counters instead of always copying.
        self.saved = tuple(np.array(a, copy=True, dtype=np.float64) for a in arrays)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        tensors = [x if isinstance(x, Tensor) else Tensor(x) for x in inputs]
        raw = [t.data for t in tensors]
        ctx = cls()
        out_data = ctx.forward(*raw)
        needs_grad = _GRAD_ENABLED and any(t.requires_grad for t in tensors)
        out = Tensor(out_data, requires_grad=needs_grad)
        if needs_grad:
            ctx.parents = tuple(tensors)
            out._ctx = ctx
        return out


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._ctx = None            # Function that produced us, or None if leaf

    # ---- Python / PyTorch conveniences --------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return Transpose.apply(self)

    @property
    def grad_fn(self):
        return self._ctx

    def __repr__(self):
        return (f"Tensor(shape={self.data.shape}, "
                f"requires_grad={self.requires_grad}, "
                f"op={type(self._ctx).__name__ if self._ctx else 'Leaf'})")

    def numpy(self):
        return self.data

    def zero_grad(self):
        self.grad = None

    def detach(self):
        """Same data, no graph. Analog of tensor.detach()."""
        return Tensor(self.data, requires_grad=False)

    # ---- The engine ---------------------------------------------------
    def backward(self, grad=None):
        """Walk the graph in reverse topological order.

        `grad` is dL/d(self). If omitted, `self` must be a scalar and we
        seed with 1 — same rule as PyTorch, because reverse-mode needs
        a vector on the left of the Jacobian.
        """
        if not self.requires_grad:
            raise RuntimeError("called backward on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs; "
                    "pass grad=... for a non-scalar tensor"
                )
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=np.float64)
            if grad.shape != self.data.shape:
                raise ValueError(
                    f"grad shape {grad.shape} does not match tensor shape {self.data.shape}"
                )

        # Depth-first post-order walk = topological order of the DAG.
        # `id(t)` because Tensor is not (and must not be) hashed by value.
        topo = []
        visited = set()

        def build(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            if t._ctx is not None:
                for p in t._ctx.parents:
                    build(p)
            topo.append(t)

        build(self)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + grad

        for t in reversed(topo):
            if t._ctx is None:
                continue                          # leaf: nothing to propagate
            grads = t._ctx.backward(t.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            if len(grads) != len(t._ctx.parents):
                raise RuntimeError(
                    f"{type(t._ctx).__name__}.backward returned {len(grads)} grads "
                    f"for {len(t._ctx.parents)} parents"
                )
            for parent, g in zip(t._ctx.parents, grads):
                if g is None or not parent.requires_grad:
                    continue
                g = unbroadcast(g, parent.data.shape)
                if parent.grad is None:
                    parent.grad = np.zeros_like(parent.data)
                # SUM RULE: a tensor used by two downstream ops gets both grads.
                parent.grad = parent.grad + g

    # ---- Ops dispatched to Function subclasses ------------------------
    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return Add.apply(other, self)

    def __sub__(self, other):
        return Sub.apply(self, other)

    def __rsub__(self, other):
        return Sub.apply(other, self)

    def __neg__(self):
        return Neg.apply(self)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(other, self)

    def __truediv__(self, other):
        return Div.apply(self, other)

    def __rtruediv__(self, other):
        return Div.apply(other, self)

    def __pow__(self, p):
        return Pow.apply(self, float(p))

    def __matmul__(self, other):
        return MatMul.apply(self, other)

    def relu(self):
        return Relu.apply(self)

    def tanh(self):
        return Tanh.apply(self)

    def exp(self):
        return Exp.apply(self)

    def log(self):
        return Log.apply(self)

    def softmax(self):
        return Softmax.apply(self)

    def sum(self):
        return Sum.apply(self)

    def mean(self):
        return Mean.apply(self)


# ----------------------------------------------------------------------
# Primitive ops. Each is a local VJP. The engine composes them.
# ----------------------------------------------------------------------

class Add(Function):
    def forward(self, a, b):
        return a + b

    def backward(self, grad):
        return grad, grad


class Sub(Function):
    def forward(self, a, b):
        return a - b

    def backward(self, grad):
        return grad, -grad


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, grad):
        return -grad


class Mul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a * b

    def backward(self, grad):
        a, b = self.saved
        return grad * b, grad * a


class Div(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a / b

    def backward(self, grad):
        a, b = self.saved
        return grad / b, -grad * a / (b ** 2)


class Pow(Function):
    def forward(self, x, p):
        # p is a Python float stored as a 0-d array by apply().
        self.p = float(np.asarray(p).reshape(()))
        self.save_for_backward(x)
        return x ** self.p

    def backward(self, grad):
        x, = self.saved
        dx = grad * self.p * (x ** (self.p - 1.0))
        return dx, None          # exponent is a constant: no grad


class MatMul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a @ b

    def backward(self, grad):
        # Same matmul rule as backprop_attention.py:
        #   Z = U @ W  =>  dU = dZ @ W^T ,  dW = U^T @ dZ
        a, b = self.saved
        return grad @ b.T, a.T @ grad


class Transpose(Function):
    def forward(self, x):
        return x.T

    def backward(self, grad):
        return grad.T


class Relu(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(x, 0.0)

    def backward(self, grad):
        x, = self.saved
        return grad * (x > 0.0)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        self.save_for_backward(y)
        return y

    def backward(self, grad):
        y, = self.saved
        return grad * (1.0 - y ** 2)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        self.save_for_backward(y)
        return y

    def backward(self, grad):
        y, = self.saved
        return grad * y          # d(e^x)/dx = e^x


class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x)

    def backward(self, grad):
        x, = self.saved
        return grad / x


class Sum(Function):
    def forward(self, x):
        self.shape = x.shape
        return np.asarray(np.sum(x), dtype=np.float64)

    def backward(self, grad):
        # Every element of x contributed equally to the sum.
        return np.ones(self.shape, dtype=np.float64) * grad


class Mean(Function):
    def forward(self, x):
        self.shape = x.shape
        self.n = x.size
        return np.asarray(np.mean(x), dtype=np.float64)

    def backward(self, grad):
        return np.ones(self.shape, dtype=np.float64) * (grad / self.n)


class Softmax(Function):
    """Row-wise softmax, axis=-1. VJP is the same formula derived in
    backprop_attention.py — we never build the (n, n) Jacobian per row:

        dS = A * (dA - sum(A * dA, axis=-1, keepdims=True))
    """

    def forward(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        a = e / np.sum(e, axis=-1, keepdims=True)
        self.save_for_backward(a)
        return a

    def backward(self, grad):
        a, = self.saved
        row_dot = np.sum(a * grad, axis=-1, keepdims=True)
        return a * (grad - row_dot)


# ======================================================================
# Finite-difference checker (same method as backprop_attention.py)
# ======================================================================

def _numerical_grad(f, x, eps=1e-6):
    """Centered finite-difference gradient of scalar f() w.r.t. array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def _check(name, loss_fn, tensors, atol=1e-5):
    for t in tensors:
        t.grad = None
    loss = loss_fn()
    loss.backward()
    ok = True
    for t in tensors:
        numeric = _numerical_grad(lambda: float(loss_fn().data), t.data)
        err = float(np.max(np.abs(t.grad - numeric)))
        status = "ok" if err < atol else "FAIL"
        if status != "ok":
            ok = False
        print(f"  [{status}] {name:18s}  max |analytic - numeric| = {err:.2e}")
    return ok


# ======================================================================
# Self-test / demos
# ======================================================================

def _demo_scalar():
    print("\n=== Part 1: scalar autograd ===")
    print("Expression:  L = (x * w + b).relu()   with x=2, w=-3, b=1")
    print("  z = x*w = -6")
    print("  y = z+b = -5")
    print("  L = relu(y) = 0     <- ReLU is off, so ALL grads are 0")

    x, w, b = Value(2.0), Value(-3.0), Value(1.0)
    L = (x * w + b).relu()
    L.backward()
    print(f"  L={L.data:.1f}  dx={x.grad:.1f}  dw={w.grad:.1f}  db={b.grad:.1f}")

    print("\nSame graph but b=10 so ReLU is on:")
    print("  y = -6+10 = 4, L=4")
    print("  dL/dy=1, dL/db=1, dL/dz=1, dL/dx=w=-3, dL/dw=x=2")
    x, w, b = Value(2.0), Value(-3.0), Value(10.0)
    L = (x * w + b).relu()
    L.backward()
    print(f"  L={L.data:.1f}  dx={x.grad:.1f}  dw={w.grad:.1f}  db={b.grad:.1f}")

    print("\nSum rule (diamond): L = x*x  =>  dL/dx = 2x")
    print("  x feeds TWO parents of the multiply (left and right).")
    print("  Each path contributes x * dL/d(x*x) = x*1, they ADD -> 2x.")
    x = Value(3.0)
    L = x * x
    L.backward()
    print(f"  x=3, L=9, dx={x.grad:.1f}  (expected 6)")


def _demo_tensor_checks():
    print("\n=== Part 2: tensor VJPs vs finite differences ===")
    rng = np.random.default_rng(0)

    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    _check("add", lambda: (a + b).sum(), [a, b])

    a = Tensor(rng.normal(size=(3, 1)), requires_grad=True)
    b = Tensor(rng.normal(size=(1, 4)), requires_grad=True)
    _check("add-broadcast", lambda: (a + b).sum(), [a, b])

    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    _check("mul", lambda: (a * b).sum(), [a, b])

    a = Tensor(rng.normal(size=(3, 5)), requires_grad=True)
    b = Tensor(rng.normal(size=(5, 2)), requires_grad=True)
    _check("matmul", lambda: (a @ b).sum(), [a, b])

    x = Tensor(rng.normal(size=(4, 6)), requires_grad=True)
    _check("relu", lambda: x.relu().sum(), [x])
    _check("tanh", lambda: x.tanh().sum(), [x])
    _check("softmax", lambda: x.softmax().sum(), [x])

    x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    _check("diamond x*x", lambda: (x * x).sum(), [x])

    x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    w = Tensor(rng.normal(size=(4, 4)), requires_grad=True)
    _check("skip-connection", lambda: (x + (x @ w).relu()).sum(), [x, w])


def _demo_mlp():
    print("\n=== Part 3: train a 2-layer MLP on XOR ===")
    rng = np.random.default_rng(1)
    X = Tensor([[0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0]])
    Y = Tensor([[0.0], [1.0], [1.0], [0.0]])

    W1 = Tensor(rng.normal(size=(2, 8)) * 0.5, requires_grad=True)
    b1 = Tensor(np.zeros(8), requires_grad=True)
    W2 = Tensor(rng.normal(size=(8, 1)) * 0.5, requires_grad=True)
    b2 = Tensor(np.zeros(1), requires_grad=True)
    params = [W1, b1, W2, b2]

    lr, steps = 0.3, 1500
    for step in range(steps + 1):
        h = (X @ W1 + b1).relu()
        pred = h @ W2 + b2
        loss = ((pred - Y) ** 2).mean()
        if step % 300 == 0:
            print(f"  step {step:4d}  loss={float(loss.data):.4f}")
        if step == steps:
            break
        for p in params:
            p.grad = None
        loss.backward()
        for p in params:
            p.data -= lr * p.grad

    with no_grad():
        h = (X @ W1 + b1).relu()
        pred = h @ W2 + b2
    print("  predictions (should be ~ [0, 1, 1, 0]):")
    print(" ", np.round(pred.data.ravel(), 3))


def _demo_attention():
    print("\n=== Part 4: attention — autograd vs hand-written backward ===")
    try:
        from backprop_attention import attention_forward, attention_backward
    except ImportError:
        print("  (skipped: could not import backprop_attention.py)")
        return

    rng = np.random.default_rng(0)
    n, d_model, dk, dv = 4, 6, 5, 3
    X = rng.normal(size=(n, d_model))
    Wq = rng.normal(size=(d_model, dk))
    Wk = rng.normal(size=(d_model, dk))
    Wv = rng.normal(size=(d_model, dv))
    dO = rng.normal(size=(n, dv))

    # Hand-written (the file you already have).
    _, cache = attention_forward(X, Wq, Wk, Wv)
    dX, dWq, dWk, dWv = attention_backward(dO, cache)

    # Same forward, but the graph is recorded automatically.
    Xt = Tensor(X, requires_grad=True)
    Wqt = Tensor(Wq, requires_grad=True)
    Wkt = Tensor(Wk, requires_grad=True)
    Wvt = Tensor(Wv, requires_grad=True)

    Q = Xt @ Wqt
    K = Xt @ Wkt
    V = Xt @ Wvt
    scale = 1.0 / np.sqrt(dk)
    S = (Q @ K.T) * scale
    A = S.softmax()
    O = A @ V
    # L = sum(O * dO)  so  dL/dO = dO, matching the hand-written setup.
    loss = (O * Tensor(dO)).sum()
    loss.backward()

    for name, analytic, auto in [
        ("X",  dX,  Xt.grad),
        ("Wq", dWq, Wqt.grad),
        ("Wk", dWk, Wkt.grad),
        ("Wv", dWv, Wvt.grad),
    ]:
        err = float(np.max(np.abs(analytic - auto)))
        status = "ok" if err < 1e-9 else "FAIL"
        print(f"  [{status}] d{name:2s}  max |hand-written - autograd| = {err:.2e}")
    print("  Same gradients. The cache/backward you wrote by hand is")
    print("  exactly what Function.save_for_backward + VJPs computed here.")


# ======================================================================
# Part 5 — Worked examples, simple -> complex
# ======================================================================

def example_1_polynomial():
    """f(x) = x^2 + 3x at x=2.  df/dx = 2x + 3 = 7."""
    print("\n[Example 1] scalar polynomial f(x) = x^2 + 3x at x=2")
    x = Value(2.0)
    f = x ** 2 + 3 * x
    f.backward()
    print(f"  f={f.data:.1f}  df/dx={x.grad:.1f}  (expected 7.0)")


def example_2_neuron():
    """One neuron: y = tanh(w1*x1 + w2*x2 + b). Hand-derived grads:
    dy/dw_i = (1 - y^2) * x_i,  dy/db = 1 - y^2."""
    print("\n[Example 2] single neuron y = tanh(w1*x1 + w2*x2 + b)")
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2, b = Value(-3.0), Value(1.0), Value(6.8813735870195432)
    y = (w1 * x1 + w2 * x2 + b).tanh()
    y.backward()
    local = 1.0 - y.data ** 2
    print(f"  y={y.data:.4f}  local tanh' = 1 - y^2 = {local:.4f}")
    print(f"  dw1={w1.grad:.4f} (expected {local * x1.data:.4f})")
    print(f"  dw2={w2.grad:.4f} (expected {local * x2.data:.4f})  <- x2=0 kills it")
    print(f"  db ={b.grad:.4f} (expected {local:.4f})")


def example_3_linear_regression_grad():
    """Tensor MSE for y_hat = X @ w.  Closed form:
    L = mean((Xw - y)^2)  =>  dL/dw = (2/N) X^T (Xw - y)."""
    print("\n[Example 3] linear regression gradient vs closed form")
    rng = np.random.default_rng(3)
    X = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 1))
    w = Tensor(rng.normal(size=(3, 1)), requires_grad=True)
    loss = ((Tensor(X) @ w - Tensor(y)) ** 2).mean()
    loss.backward()
    closed = (2.0 / X.shape[0]) * X.T @ (X @ w.data - y)
    err = float(np.max(np.abs(w.grad - closed)))
    print(f"  loss={float(loss.data):.4f}  max |autograd - closed form| = {err:.2e}"
          f"  [{'ok' if err < 1e-12 else 'FAIL'}]")


def example_4_weight_sharing():
    """Sum rule with tensors: the SAME W is used in two layers.
    h = relu(x @ W);  out = h @ W;  L = sum(out).
    dW must be the sum of the grads from both uses — the engine does
    this automatically by accumulating into W.grad."""
    print("\n[Example 4] weight sharing (one W used twice) — sum rule")
    rng = np.random.default_rng(4)
    x = Tensor(rng.normal(size=(2, 4)))
    W = Tensor(rng.normal(size=(4, 4)), requires_grad=True)
    _check("shared W", lambda: ((x @ W).relu() @ W).sum(), [W])

    # Make the two contributions explicit by using two independent copies.
    W1 = Tensor(W.data.copy(), requires_grad=True)
    W2 = Tensor(W.data.copy(), requires_grad=True)
    ((x @ W1).relu() @ W2).sum().backward()
    err = float(np.max(np.abs(W.grad - (W1.grad + W2.grad))))
    print(f"  dW == dW_layer1 + dW_layer2 ?  max err = {err:.2e}")


def example_5_stop_gradient():
    """detach() and no_grad() cut the graph.
    L = sum(a * b.detach())  =>  dL/da = b,  b gets no gradient.
    Used for target networks, EMA teachers, straight-through tricks."""
    print("\n[Example 5] stop-gradient with detach() and no_grad()")
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    (a * b.detach()).sum().backward()
    print(f"  da={a.grad}  (expected b = [4, 5, 6])")
    print(f"  db={b.grad}  (expected None — detached)")

    with no_grad():
        c = a * b
    print(f"  under no_grad: c.requires_grad={c.requires_grad}, grad_fn={c.grad_fn}")


def example_6_softmax_cross_entropy():
    """Softmax + cross-entropy built from primitives (softmax, log, mul,
    mean). The fused closed form is the famous:
        dL/dlogits = (softmax(logits) - onehot) / N
    Autograd recovers it without being told."""
    print("\n[Example 6] softmax cross-entropy: dlogits = (p - onehot) / N")
    rng = np.random.default_rng(6)
    N, C = 5, 4
    logits = Tensor(rng.normal(size=(N, C)), requires_grad=True)
    labels = rng.integers(0, C, size=N)
    onehot = np.eye(C)[labels]

    p = logits.softmax()
    loss = -(Tensor(onehot) * p.log()).sum() * (1.0 / N)
    loss.backward()

    closed = (p.data - onehot) / N
    err = float(np.max(np.abs(logits.grad - closed)))
    print(f"  loss={float(loss.data):.4f}  max |autograd - closed form| = {err:.2e}"
          f"  [{'ok' if err < 1e-12 else 'FAIL'}]")


if __name__ == "__main__":
    _demo_scalar()
    _demo_tensor_checks()
    _demo_mlp()
    _demo_attention()
    example_1_polynomial()
    example_2_neuron()
    example_3_linear_regression_grad()
    example_4_weight_sharing()
    example_5_stop_gradient()
    example_6_softmax_cross_entropy()
    print("\nDone. Open autograd.py and read it top to bottom — the comments")
    print("are the lesson; the code is the engine.")
