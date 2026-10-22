import sys, torch
from env import MicrogridEnv
from dqn import QNet
from baselines import idle_policy, rule_policy, make_scenarios, evaluate, evaluate_oracle

def load_policy(path):
    q = QNet(11, 6, dueling=any(k in path for k in ("dueling", "full", "constrained"))) #q = QNet(11, 6)

    q.load_state_dict(torch.load(path)); q.eval()
    def pol(obs):
        with torch.no_grad():
            return int(q(torch.from_numpy(obs).unsqueeze(0)).argmax())
    return pol

val_env = MicrogridEnv("data/solar_hourly.csv", years=[2022], seed=0)
sc = make_scenarios(val_env, n=100, seed=7)          # same exam as training validation
rows = {"idle": evaluate(val_env, idle_policy, sc),
        "rule": evaluate(val_env, rule_policy, sc),
        "oracle": evaluate_oracle(val_env, sc)}
for path in sys.argv[1:]:
    rows[path] = evaluate(val_env, load_policy(path), sc)

for k, v in rows.items():
    line = f"{k:30s} return {v['return']:7.2f}  cost {v['cost_Rs']:8.0f}  unserved {v['unserved_crit_kWh']:5.1f}"
    if k not in ("idle", "rule", "oracle"):
        gap = (v["return"] - rows["rule"]["return"]) / (rows["oracle"]["return"] - rows["rule"]["return"])
        line += f"  | gap closed vs rule->oracle: {100*gap:5.1f}%"
    print(line)