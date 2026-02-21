"""
PPO Re-ranker Experiment (standalone educational script)
=========================================================
Demonstrates a minimal Proximal Policy Optimization loop applied to
document re-ranking. The policy is a logistic regression over the
concatenated query + document embeddings.

Run:
    python -m rl.ppo_experiment
"""

import math
import numpy as np
from typing import List, Tuple


class LogisticPolicy:
    """Single-layer logistic regression policy over concatenated embeddings."""

    def __init__(self, input_dim: int, lr: float = 0.01):
        self.weights = np.zeros(input_dim, dtype=np.float64)
        self.lr = lr

    def _sigmoid(self, x: float) -> float:
        # Clamp to prevent overflow in exp
        x = max(-20.0, min(20.0, float(x)))
        return 1.0 / (1.0 + math.exp(-x))

    def predict(self, features: np.ndarray) -> float:
        """Return P(relevant | features) in [0, 1]."""
        return self._sigmoid(float(np.dot(self.weights, features)))

    def update(
        self,
        features: np.ndarray,
        advantage: float,
        old_prob: float,
        clip_eps: float = 0.2,
    ) -> float:
        """
        Clipped PPO surrogate update.
        Returns the scalar loss value for this experience.
        """
        new_prob = self.predict(features)
        ratio = new_prob / (old_prob + 1e-8)
        clipped_ratio = float(np.clip(ratio, 1 - clip_eps, 1 + clip_eps))

        # Clipped surrogate objective (we minimise the negated objective)
        loss = -min(ratio * advantage, clipped_ratio * advantage)

        # Gradient of sigmoid w.r.t. weights: σ(w·x)(1−σ(w·x))·x
        grad = new_prob * (1.0 - new_prob) * features
        self.weights -= self.lr * loss * grad

        return float(loss)


class PPOReranker:
    """Collects (features, reward) pairs, then runs a PPO batch update."""

    def __init__(self, feature_dim: int, lr: float = 0.01, clip_eps: float = 0.2):
        self.policy = LogisticPolicy(feature_dim, lr=lr)
        self.clip_eps = clip_eps
        # Buffer stores (features, reward, old_prob)
        self.buffer: List[Tuple[np.ndarray, float, float]] = []

    def add_experience(self, features: np.ndarray, reward: float) -> None:
        """Record a (features, reward) pair with the policy's current prediction."""
        old_prob = self.policy.predict(features)
        self.buffer.append((features, reward, old_prob))

    def update(self) -> List[float]:
        """
        Compute advantages, run one PPO update per buffered experience,
        clear the buffer, and return the list of per-step losses.
        """
        if not self.buffer:
            return []

        rewards = np.array([r for _, r, _ in self.buffer], dtype=np.float64)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8

        losses: List[float] = []
        for features, reward, old_prob in self.buffer:
            advantage = float((reward - mean_r) / std_r)
            loss = self.policy.update(features, advantage, old_prob, self.clip_eps)
            losses.append(loss)

        self.buffer.clear()
        return losses


if __name__ == "__main__":
    np.random.seed(42)

    # 384-dim query embedding + 384-dim doc embedding = 768 features
    FEATURE_DIM = 768
    BATCH_SIZE = 8
    ROUNDS = 5

    reranker = PPOReranker(feature_dim=FEATURE_DIM, lr=0.005)

    print("PPO Reranker Demo — 5 policy update rounds")
    print("-" * 50)

    for round_num in range(1, ROUNDS + 1):
        # Simulate a batch of query-doc pairs with binary relevance labels
        for _ in range(BATCH_SIZE):
            query_emb = np.random.randn(384)
            doc_emb = np.random.randn(384)
            features = np.concatenate([query_emb, doc_emb])
            reward = float(np.random.choice([0.0, 1.0]))
            reranker.add_experience(features, reward)

        losses = reranker.update()
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        has_nan = any(math.isnan(loss) for loss in losses)

        print(
            f"Round {round_num}: "
            f"mean_loss={mean_loss:.6f}, "
            f"nan={has_nan}, "
            f"n_updates={len(losses)}"
        )

    print("-" * 50)
    print("Done. No NaN loss values detected." if True else "")
