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

# state_dict[i][j] = how many times token j follows token i
state_dict: list[list[int]] = [[0] * vocab_size for _ in range(vocab_size)]


# model
def bigram(token_id: int) -> list[float]:
    row: list[int] = state_dict[token_id]
    total: int = sum(row) + vocab_size  # additive smoothing
    return [(c + 1) / total for c in row]


# train
def train_model() -> None:
    num_steps = 1000
    min_loss: float = 1e3

    for step in range(num_steps):
        # take a token
        doc: str = docs[step % len(docs)]
        tokens: list[int] = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n: int = len(tokens) - 1

        # forward pass: compute loss
        losses: list[float] = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            probs = bigram(token_id)
            loss_t = -math.log(probs[target_id])
            losses.append(loss_t)
        loss = 1 / n * sum(losses)  # average
        min_loss = min(min_loss, loss)

        # update model
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            state_dict[token_id][target_id] += 1

        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss:.4f}")
    print(f"min loss: {min_loss:.4f}")


# inference: sample new names
def inference() -> None:
    print("\n--- inference (new, hallucinated names) ---")
    for sample_id in range(20):
        token_id = BOS
        sample: list[str] = []

        for _ in range(16):
            token_id: int = random.choices(range(vocab_size), weights=bigram(token_id))[
                0
            ]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_id + 1:2d} : {''.join(sample)}")


train_model()
inference()
