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
        eps: float = 1e-6
        return Value(math.log(self.data + eps), (self,), (1 / (self.data + eps),))

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
def matrix(n_out: int, n_in: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(n_in)] for _ in range(n_out)]


n_embd: int = 16
n_head: int = 4
n_layer: int = 1
block_size: int = 16
head_dim: int = n_embd // n_head
state_dict: dict[str, list[list[Value]]] = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}

for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

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


def rmsnorm(x: list[Value]) -> list[Value]:
    ms: Value = sum((xi * xi for xi in x), start=Value(0.0)) / len(x)
    scale: Value = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
) -> list[Value]:
    # embeddings
    tok_emb: list[Value] = state_dict["wte"][token_id]
    pos_emb: list[Value] = state_dict["wpe"][pos_id]
    x: list[Value] = [t + p for t, p in zip(tok_emb, pos_emb)]

    for i in range(n_layer):
        x_residual: list[Value] = x
        x = rmsnorm(x)
        q: list[Value] = linear(x, state_dict[f"layer{i}.attn_wq"])
        k: list[Value] = linear(x, state_dict[f"layer{i}.attn_wk"])
        v: list[Value] = linear(x, state_dict[f"layer{i}.attn_wv"])
        keys[i].append(k)
        values[i].append(v)

        # multi-head attention block
        x_attn: list[Value] = []
        for h in range(n_head):
            hs = h * head_dim  # head_start
            q_h: list[Value] = q[hs : hs + head_dim]
            keys_h: list[list[Value]] = [ki[hs : hs + head_dim] for ki in keys[i]]
            values_h: list[list[Value]] = [vi[hs : hs + head_dim] for vi in values[i]]
            attn_logits: list[Value] = [
                sum(q_h[c] * keys_h[r][c] for c in range(head_dim)) / head_dim**0.5
                for r in range(len(keys_h))
            ]
            attn_weights: list[Value] = softmax(attn_logits)
            head_out: list[Value] = [
                sum(
                    (attn_weights[r] * values_h[r][c] for r in range(len(values_h))),
                    start=Value(0.0),
                )
                for c in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{i}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        # multi-layer perceptron block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{i}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{i}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits: list[Value] = linear(x, state_dict["lm_head"])
    return logits


# train
def train_model() -> None:
    num_steps: int = 1000
    learning_rate: float = 0.01
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    m: list[float] = [0.0] * len(params)  # first moment buffer
    v: list[float] = [0.0] * len(params)  # second moment buffer
    min_loss: float = 1e3

    for step in range(num_steps):
        # take a token
        doc: str = docs[step % len(docs)]
        tokens: list[int] = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n: int = min(block_size, len(tokens) - 1)

        # forward
        losses: list[Value] = []
        keys: list[list[list[Value]]] = [[] for _ in range(n_layer)]
        values: list[list[list[Value]]] = [[] for _ in range(n_layer)]
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits: list[Value] = gpt(token_id, pos_id, keys, values)
            probs: list[Value] = softmax(logits)
            loss_t: Value = -probs[target_id].log()
            losses.append(loss_t)
        loss: Value = (1.0 / n) * sum(losses, start=Value(0.0))
        min_loss = min(min_loss, loss.data)

        # backward
        loss.backward()

        # gradient clipping -> limit speed
        max_norm: float = 1.0
        global_norm: float = math.sqrt(sum(p.grad**2 for p in params))

        if global_norm > max_norm:
            clip_coef: float = max_norm / (global_norm + 1e-6)
            for p in params:
                p.grad *= clip_coef

        # adam update
        lr_t: float = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat: float = m[i] / (1 - beta1 ** (step + 1))  # bias correction
            v_hat: float = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
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
        keys: list[list[list[Value]]] = [[] for _ in range(n_layer)]
        values: list[list[list[Value]]] = [[] for _ in range(n_layer)]

        for pos_id in range(block_size):
            logits: list[Value] = gpt(token_id, pos_id, keys, values)
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
