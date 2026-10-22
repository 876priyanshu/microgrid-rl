import sys, torch, numpy as np
from env import MicrogridEnv
from dqn import QNet
from baselines import rule_policy, make_scenarios, evaluate, evaluate_oracle


suffix = "_best" if "--best" in sys.argv else ""
names = ["vanilla", "double", "dueling", "per", "full", "constrained"]
seeds = [0, 1, 2]
test_env = MicrogridEnv("data/solar_hourly.csv", years=[2023], seed=0)   # held-out TEST year
sc = make_scenarios(test_env, n=200, seed=11)

def load_policy(path, dueling):
    q = QNet(11, 6, dueling=dueling)
    q.load_state_dict(torch.load(path)); q.eval()
    def pol(obs):
        with torch.no_grad():
            return int(q(torch.from_numpy(obs).unsqueeze(0)).argmax())
    return pol


rule = evaluate(test_env, rule_policy, sc)
oracle = evaluate_oracle(test_env, sc)
print(f"rule   return {rule['return']:.2f}  cost {rule['cost_Rs']:.0f}  unserved {rule['unserved_crit_kWh']:.2f}")
print(f"oracle return {oracle['return']:.2f}  cost {oracle['cost_Rs']:.0f}  unserved {oracle['unserved_crit_kWh']:.2f}\n")
span = oracle["return"] - rule["return"]

for name in names:
    dueling = name in ("dueling", "full", "constrained")
    rs, us, cs = [], [], []
    for s in seeds:
        try:
            res = evaluate(test_env, load_policy(f"results/{name}_s{s}{suffix}.pt", dueling), sc)
        except FileNotFoundError:
            continue
        rs.append(res["return"]); us.append(res["unserved_crit_kWh"]); cs.append(res["cost_Rs"])
    if rs:
        gap = [100 * (x - rule["return"]) / span for x in rs]
        print(f"{name:12s} return {np.mean(rs):6.2f}+-{np.std(rs):4.2f} | cost {np.mean(cs):6.0f} | "
              f"unserved {np.mean(us):4.2f}+-{np.std(us):4.2f} | gap closed {np.mean(gap):5.1f}%+-{np.std(gap):3.1f} (n={len(rs)})")