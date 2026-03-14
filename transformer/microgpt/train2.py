# https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a

from __future__ import annotations

import math
import os
import random

random.seed(42)

docs: list[str] = []

# dataset
if not os.path.exists("input.txt"):
    import urllib.request

    name_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    _ = urllib.request.urlretrieve(name_url, "input.txt")

docs = [
    line.strip()
    for line in open("input.txt").read().strip().split("\n")
    if line.strip()
]
random.shuffle(docs)

print(f"num docs: {len(docs)}")
print(f"few names: {docs[:5]}")

# tokenizer
uchars: list[str] = sorted(set("".join(docs)))
BOS: int = len(uchars)
vocab_size: int = len(uchars) + 1
print(f"vocab size: {vocab_size}")


# autograd
class Value:
    __slots__: tuple[str, ...] = ("grad", "data", "_children", "_local_grads")
    # children: (x, y, ...)
    # local_grads: (d/dx, d/dy, ...)

    grad: float
    data: float
    _children: tuple[Value, ...]
    _local_grads: tuple[float, ...]

    def __init__(
        self,
        data: float,
        children: tuple[Value, ...] = (),
        _local_grads: tuple[float, ...] = (),
    ):
        self.grad = 0
        self.data = data
        self._children = children
        self._local_grads = _local_grads
        pass

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other: Value | int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __rmul__(self, other: Value | int | float) -> Value:
        return self * other

    def __pow__(self, other: int | float) -> Value:
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self) -> Value:
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self) -> Value:
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | int | float) -> Value:
        return other + (-self)

    def __truediv__(self, other: Value | int | float) -> Value:
        return self * (other**-1)

    def __rtruediv__(self, other: Value | int | float) -> Value:
        return other * (self**-1)

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)  # build computational graph
        self.grad = 1  # dL/dL = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += v.grad * local_grad  # dL/dx = dL/df * df/dx


# parameters
def matrix(n_out: int, n_in: int) -> list[list[Value]]:
    return [[Value(random.gauss(0, 0.08)) for _ in range(n_in)] for _ in range(n_out)]


n_embd: int = 16
state_dict: dict[str, list[list[Value]]] = {
    "wte": matrix(vocab_size, n_embd),
    "mlp_fc1": matrix(4 * n_embd, n_embd),
    "mlp_fc2": matrix(vocab_size, 4 * n_embd),
}
params: list[Value] = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")


# model
def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum((wi * xi for wi, xi in zip(wo, x)), start=Value(0.0)) for wo in w]


def softmax(logits: list[Value]) -> list[Value]:
    max_value: float = max(val.data for val in logits)
    exps: list[Value] = [(val - max_value).exp() for val in logits]
    total: Value = sum(exps, start=Value(0.0))
    props: list[Value] = [e / total for e in exps]
    return props


def mlp(token_id: int) -> list[Value]:
    x: list[Value] = state_dict["wte"][token_id]
    x = linear(x, state_dict["mlp_fc1"])
    x = [xi.relu() for xi in x]
    logits: list[Value] = linear(x, state_dict["mlp_fc2"])
    return logits


# train
def train_model() -> None:
    num_steps: int = 1000
    learning_rate: float = 0.5
    min_loss: float = 1e3

    for step in range(num_steps):
        # take a token
        doc: str = docs[step % len(docs)]
        tokens: list[int] = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n: int = len(tokens) - 1

        # forward
        losses: list[Value] = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits: list[Value] = mlp(token_id)
            probs: list[Value] = softmax(logits)
            loss_t: Value = -probs[target_id].log()
            losses.append(loss_t)
        loss: Value = (1.0 / n) * sum(losses, start=Value(0.0))
        min_loss = min(min_loss, loss.data)

        # backward
        loss.backward()

        # sgd update
        lr_t: float = learning_rate * (1 - step / num_steps)
        for _, p in enumerate(params):
            p.data -= lr_t * p.grad
            p.grad = 0

        if (step < 5) or (step % 200) == 0:
            print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    print(f"min loss: {min_loss:.4f}")


# inference: sample new names
def inference() -> None:
    temperature: float = 0.5
    print("\n--- inference (new, hallucinated names) ---")

    for sample_id in range(20):
        token_id: int = BOS
        sample: list[str] = []

        for pos_id in range(16):
            logits: list[Value] = mlp(token_id)
            probs: list[Value] = softmax([val / temperature for val in logits])
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_id + 1:2d} : {''.join(sample)}")


train_model()
inference()
