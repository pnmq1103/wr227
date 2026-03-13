# https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a

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


# parameters
def matrix(n_out: int, n_in: int) -> list[list[float]]:
    return [[random.gauss(0, 0.08) for _ in range(n_in)] for _ in range(n_out)]


n_embd: int = 16
state_dict: dict[str, list[list[float]]] = {
    "wte": matrix(vocab_size, n_embd),
    "mlp_fc1": matrix(4 * n_embd, n_embd),
    "mlp_fc2": matrix(vocab_size, 4 * n_embd),
}
params: list[tuple[list[float], int]] = [
    (row, j) for mat in state_dict.values() for row in mat for j in range(len(row))
]
print(f"num params: {len(params)}")


# model
def linear(x: list[float], w: list[list[float]]) -> list[float]:
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits: list[float]) -> list[float]:
    max_value: float = max(logits)
    exps: list[float] = [math.exp(val - max_value) for val in logits]
    total: float = sum(exps)
    probs = [e / total for e in exps]
    return probs


def relu(x: list[float]) -> list[float]:
    return [max(0, xi) for xi in x]


def mlp(token_id: int) -> list[float]:
    x: list[float] = state_dict["wte"][token_id]
    x = linear(x, state_dict["mlp_fc1"])
    x = relu(x)
    logits: list[float] = linear(x, state_dict["mlp_fc2"])
    return logits


# forward pass
def forward(tokens: list[int], n: int) -> float:
    losses: list[float] = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = mlp(token_id)
        probs = softmax(logits)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    return loss


def numerical_gradient(tokens: list[int], n: int) -> tuple[float, list[float]]:
    loss: float = forward(tokens, n)
    eps: float = 1e-5
    grad: list[float] = []
    for mat in state_dict.values():
        for row in mat:
            for j in range(len(row)):
                old: float = row[j]
                row[j] += eps
                loss_plus: float = forward(tokens, n)
                row[j] = old
                grad.append((loss_plus - loss) / eps)
    return loss, grad


# backpropagation
def analytic_gradient(tokens: list[int], n: int) -> tuple[float, list[float]]:
    grad: dict[str, list[list[float]]] = {
        k: [[0.0] * len(row) for row in mat] for k, mat in state_dict.items()
    }
    losses: list[float] = []

    for pos_id in range(n):
        # forward pass
        # x = wte[token_id]
        # h_pre = W_1 @ x
        # h = relu(h_pre)
        # logits = W_2 @ h
        # probs = softmax(logits)
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        x: list[float] = state_dict["wte"][token_id]
        hidden_pre: list[float] = linear(x, state_dict["mlp_fc1"])
        hidden: list[float] = relu(hidden_pre)
        logits: list[float] = linear(hidden, state_dict["mlp_fc2"])
        probs: list[float] = softmax(logits)
        loss_t: float = -math.log(probs[target_id])
        losses.append(loss_t)

        # backward pass
        # d(loss)/d(logits)
        # for softmax + cross-entropy = probs - kronecker(target), divide n to average
        dlogits: list[float] = [p / n for p in probs]  # dl/dz2
        dlogits[target_id] -= 1.0 / n  # kronecker

        # d(loss)/d(mlp_fc2), d(loss)/d(h)
        dh: list[float] = [0.0] * len(hidden)
        for i in range(len(dlogits)):
            for j in range(len(hidden)):
                # dL/W_2 = dL/dz2 * dz2/dW2 = dL/dz2 * h
                grad["mlp_fc2"][i][j] += dlogits[i] * hidden[j]
                # dL/dh = dL/dz2 * dz2/dh = W2^T @ dl/dz2
                dh[j] += state_dict["mlp_fc2"][i][j] * dlogits[i]

        # d(loss)/d(h_pre)
        # dL/dh_pre = dL/dh * dh/dh_pre = dL/dh * relu'(h_pre)
        dh_pre: list[float] = [
            dh[j] * (1.0 if hidden_pre[j] > 0 else 0.0) for j in range(len(hidden_pre))
        ]

        # d(loss)/d(mlp_fc1), d(loss)/d(x)
        dx: list[float] = [0.0] * len(x)
        for i in range(len(dh_pre)):
            for j in range(len(x)):
                grad["mlp_fc1"][i][j] += dh_pre[i] * x[j]
                dx[j] += state_dict["mlp_fc1"][i][j] * dh_pre[i]

        # d(loss)/d(wte[token_id])
        for j in range(len(x)):
            grad["wte"][token_id][j] += dx[j]

    loss: float = (1.0 / n) * sum(losses)
    grad_flat: list[float] = [g for mat in grad.values() for row in mat for g in row]
    return loss, grad_flat


# gradient check
doc = docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = len(tokens) - 1
loss_n, grad_n = numerical_gradient(tokens, n)
loss_a, grad_a = analytic_gradient(tokens, n)
grad_diff = max(abs(gn - ga) for gn, ga in zip(grad_n, grad_a))
print(
    f"gradient check | loss_n {loss_n:.6f} | loss_a {loss_a:.6f} | max diff {grad_diff:.8f}"
)


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

        # forward & backward
        loss, grad = analytic_gradient(tokens, n)
        min_loss = min(min_loss, loss)

        # sgd update
        lr_t: float = learning_rate * (1 - step / num_steps)
        for i, (row, j) in enumerate(params):
            row[j] -= lr_t * grad[i]

        if (step < 5) or (step % 200) == 0:
            print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss:.4f}")

    print(f"min loss: {min_loss:.4f}")


# inference: sample new names
def inference() -> None:
    temperature: float = 0.5
    print("\n--- inference (new, hallucinated names) ---")
    for sample_id in range(20):
        token_id: int = BOS
        sample: list[str] = []

        for pos_id in range(16):
            logits: list[float] = mlp(token_id)
            probs: list[float] = softmax([lo / temperature for lo in logits])
            token_id = random.choices(range(vocab_size), weights=probs)[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_id + 1:2d} : {''.join(sample)}")


train_model()
inference()
