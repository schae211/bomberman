
"""
Utility script to check what the agent is doing is a given game round
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

with open("stats.json") as f:
    data = json.load(f)

by_round = data["by_round"]

results = pd.DataFrame({
    "round": by_round.keys(),
    "coins": [val["coins"] for _, val in by_round.items()],
    "kills": [val["kills"] for _, val in by_round.items()],
    "steps": [val["steps"] for _, val in by_round.items()],
    "suicides": [val["suicides"] for _, val in by_round.items()]
})

plt.plot(results.coins)
plt.xlabel("Game")
plt.ylabel("Total Coins")
plt.savefig("coin_evaluation.pdf")
