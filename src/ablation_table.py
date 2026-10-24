import sys, json, torch, numpy as np
from env import MicrogridEnv
from dqn import QNet
from baselines import rule_policy, make_scenarios, evaluate, evaluate_oracle

suffix = "_best" if "--best" in sys.argv else ""
names = ["vanilla", "double", "dueling", "per", "full", "constrained", "constrained_v2"]
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

rule = {k: float(v) for k, v in evaluate(test_env, rule_policy, sc).items()}
oracle = {k: float(v) for k, v in evaluate_oracle(test_env, sc).items()}
print(f"rule   return {rule['return']:.2f}  cost {rule['cost_Rs']:.0f}  unserved {rule['unserved_crit_kWh']:.2f}")
print(f"oracle return {oracle['return']:.2f}  cost {oracle['cost_Rs']:.0f}  unserved {oracle['unserved_crit_kWh']:.2f}\n")
table = {"rule": rule, "oracle": oracle}
span = oracle["return"] - rule["return"]
for name in names:
    dueling = any(k in name for k in ("dueling", "full", "constrained"))
    rs, us, cs = [], [], []
    for s in seeds:
        try:
            res = evaluate(test_env, load_policy(f"results/{name}_s{s}{suffix}.pt", dueling), sc)
        except FileNotFoundError:
            continue
        rs.append(res["return"]); us.append(res["unserved_crit_kWh"]); cs.append(res["cost_Rs"])
    if rs:
        gap = [100 * (x - rule["return"]) / span for x in rs]
        table[name] = {"return_mean": float(np.mean(rs)), "return_std": float(np.std(rs)),
                       "cost_mean": float(np.mean(cs)),
                       "unserved_mean": float(np.mean(us)), "unserved_std": float(np.std(us)),
                       "gap_mean": float(np.mean(gap)), "gap_std": float(np.std(gap))}
        print(f"{name:15s} return {np.mean(rs):6.2f}+-{np.std(rs):4.2f} | cost {np.mean(cs):6.0f} | "
              f"unserved {np.mean(us):4.2f}+-{np.std(us):4.2f} | gap closed {np.mean(gap):5.1f}%+-{np.std(gap):3.1f} (n={len(rs)})")
json.dump(table, open(f"results/ablation_test{suffix}.json", "w"), indent=2)