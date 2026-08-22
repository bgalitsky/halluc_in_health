import numpy as np
import random

ACTIONS = [
    "public_goods",
    "market_reform",
    "corruption",
    "censorship",
    "repression",
    "external_enemy",
    "fair_election",
    "election_manipulation"
]

class RegimeEnv:
    def __init__(self, regime="democracy"):
        self.regime = regime
        self.reset()

    def reset(self):
        self.welfare = 0.5
        self.gdp = 0.5
        self.inflation = 0.3
        self.corruption = 0.4
        self.rule_of_law = 0.5
        self.info_freedom = 0.5
        self.opposition = 0.5
        self.fear = 0.2
        self.elite_loyalty = 0.5
        self.election_quality = 0.6 if self.regime == "democracy" else 0.2
        return self.state()

    def state(self):
        values = [
            self.welfare,
            self.gdp,
            self.inflation,
            self.corruption,
            self.rule_of_law,
            self.info_freedom,
            self.opposition,
            self.fear,
            self.elite_loyalty,
            self.election_quality
        ]
        # Discretize into low/medium/high bins
        return tuple(int(min(2, max(0, v * 3))) for v in values)

    def step(self, action):
        if action == "public_goods":
            self.welfare += 0.08
            self.gdp += 0.05
            self.inflation -= 0.02

        elif action == "market_reform":
            self.gdp += 0.08
            self.rule_of_law += 0.04
            self.corruption -= 0.03

        elif action == "corruption":
            self.corruption += 0.08
            self.elite_loyalty += 0.05
            self.gdp -= 0.04
            self.welfare -= 0.04

        elif action == "censorship":
            self.info_freedom -= 0.10
            self.opposition -= 0.08
            self.gdp -= 0.03

        elif action == "repression":
            self.fear += 0.10
            self.opposition -= 0.10
            self.rule_of_law -= 0.08
            self.gdp -= 0.03

        elif action == "external_enemy":
            self.fear += 0.06
            self.elite_loyalty += 0.04
            self.inflation += 0.05
            self.gdp -= 0.05
            self.welfare -= 0.05

        elif action == "fair_election":
            self.election_quality += 0.08
            self.rule_of_law += 0.05
            self.info_freedom += 0.04

        elif action == "election_manipulation":
            self.election_quality -= 0.10
            self.rule_of_law -= 0.06
            self.opposition -= 0.04
            self.corruption += 0.04

        self.clip()

        if self.regime == "democracy":
            survival = (
                0.35 * self.welfare
                + 0.30 * self.gdp
                + 0.20 * self.rule_of_law
                + 0.20 * self.election_quality
                - 0.30 * self.corruption
                - 0.25 * self.inflation
            )
        else:
            survival = (
                0.30 * self.fear
                + 0.25 * self.elite_loyalty
                + 0.25 * (1 - self.info_freedom)
                + 0.25 * (1 - self.opposition)
                - 0.20 * self.inflation
                - 0.15 * self.gdp
            )

        # Add long-run social welfare penalty to show national cost
        social_cost = (
            -0.20 * self.corruption
            -0.15 * self.inflation
            +0.20 * self.gdp
            +0.20 * self.welfare
            +0.10 * self.rule_of_law
        )

        reward = survival + social_cost
        done = False
        return self.state(), reward, done

    def clip(self):
        for name in [
            "welfare", "gdp", "inflation", "corruption", "rule_of_law",
            "info_freedom", "opposition", "fear", "elite_loyalty",
            "election_quality"
        ]:
            setattr(self, name, min(1.0, max(0.0, getattr(self, name))))


def train(regime="democracy", episodes=5000, steps=50, alpha=0.1, gamma=0.95, eps=0.1):
    env = RegimeEnv(regime)
    Q = {}

    for _ in range(episodes):
        s = env.reset()
        for _ in range(steps):
            if s not in Q:
                Q[s] = np.zeros(len(ACTIONS))

            if random.random() < eps:
                a_idx = random.randrange(len(ACTIONS))
            else:
                a_idx = int(np.argmax(Q[s]))

            s2, r, _ = env.step(ACTIONS[a_idx])

            if s2 not in Q:
                Q[s2] = np.zeros(len(ACTIONS))

            Q[s][a_idx] += alpha * (r + gamma * np.max(Q[s2]) - Q[s][a_idx])
            s = s2

    return Q


def simulate_policy(regime, Q, steps=30):
    env = RegimeEnv(regime)
    s = env.reset()
    history = []

    for _ in range(steps):
        if s not in Q:
            action = random.choice(ACTIONS)
        else:
            action = ACTIONS[int(np.argmax(Q[s]))]

        s, reward, _ = env.step(action)
        history.append({
            "action": action,
            "welfare": env.welfare,
            "gdp": env.gdp,
            "inflation": env.inflation,
            "corruption": env.corruption,
            "rule_of_law": env.rule_of_law,
            "info_freedom": env.info_freedom,
            "opposition": env.opposition,
            "fear": env.fear,
            "elite_loyalty": env.elite_loyalty,
            "election_quality": env.election_quality,
            "reward": reward
        })

    return history


Q_demo = train("democracy")
Q_auth = train("authoritarian")

demo_path = simulate_policy("democracy", Q_demo)
auth_path = simulate_policy("authoritarian", Q_auth)

print("Democratic learned actions:")
print([x["action"] for x in demo_path[:15]])

print("\nAuthoritarian learned actions:")
print([x["action"] for x in auth_path[:15]])