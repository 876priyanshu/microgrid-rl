from env import MicrogridEnv
from baselines import idle_policy, rule_policy, make_scenarios, evaluate, evaluate_oracle

env = MicrogridEnv("data/solar_hourly.csv", years=[2023], seed=0)   # held-out year
sc = make_scenarios(env, n=60)
print("idle  ", evaluate(env, idle_policy, sc))
print("rule  ", evaluate(env, rule_policy, sc))
print("oracle", evaluate_oracle(env, sc))