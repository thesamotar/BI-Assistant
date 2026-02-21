import math
import json
from typing import Dict, Any


class UCB1Bandit:
    """
    UCB1 multi-armed bandit for re-ranking documents based on user feedback.

    Each 'arm' is a document URL. Rewards are 1.0 (positive feedback) or
    0.0 (negative feedback). The UCB1 score balances exploitation (mean
    reward) with exploration (uncertainty bonus).
    """

    def __init__(self):
        self._arms: Dict[str, Dict[str, float]] = {}
        self._total_pulls: int = 0

    def update(self, arm_id: str, reward: float) -> None:
        """Record a reward observation for the given arm."""
        if arm_id not in self._arms:
            self._arms[arm_id] = {"pulls": 0, "total_reward": 0.0}
        self._arms[arm_id]["pulls"] += 1
        self._arms[arm_id]["total_reward"] += reward
        self._total_pulls += 1

    def get_score(self, arm_id: str) -> float:
        """
        Return the UCB1 score for an arm:
            score = mean_reward + sqrt(2 * ln(N) / n)

        Returns 0.0 for arms that have never been pulled.
        """
        if arm_id not in self._arms or self._arms[arm_id]["pulls"] == 0:
            return 0.0

        arm = self._arms[arm_id]
        mean_reward = arm["total_reward"] / arm["pulls"]

        if self._total_pulls <= 1:
            return mean_reward

        exploration = math.sqrt(2 * math.log(self._total_pulls) / arm["pulls"])
        return mean_reward + exploration

    def load_from_supabase(self, client: Any, feedback_table: str) -> None:
        """Rebuild bandit state from all historical feedback stored in Supabase."""
        try:
            page_size = 1000
            offset = 0
            count = 0
            while True:
                response = (
                    client.table(feedback_table)
                    .select("sources, feedback")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                rows = response.data
                if not rows:
                    break
                for row in rows:
                    sources = row.get("sources") or []
                    feedback = row.get("feedback", "")
                    reward = 1.0 if feedback == "positive" else 0.0
                    for url in sources:
                        self.update(url, reward)
                    count += 1
                offset += page_size
                if len(rows) < page_size:
                    break
            print(f"[UCB1Bandit] Loaded {count} feedback records from Supabase")
        except Exception as e:
            # Table may not exist yet on first startup — that's fine
            print(f"[UCB1Bandit] Could not load feedback from Supabase: {e}")

    def save_to_json(self, path: str) -> None:
        """Persist bandit state to a JSON file (debug helper)."""
        data = {"arms": self._arms, "total_pulls": self._total_pulls}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[UCB1Bandit] Saved state to {path}")

    def load_from_json(self, path: str) -> None:
        """Restore bandit state from a JSON file (debug helper)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._arms = data.get("arms", {})
        self._total_pulls = data.get("total_pulls", 0)
        print(f"[UCB1Bandit] Loaded state from {path} ({len(self._arms)} arms)")
