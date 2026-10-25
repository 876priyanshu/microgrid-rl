# Run a trained agent on one held-out test day and print its hourly decisions next to the rule baseline.
# Usage: python src/demo.py [model_path] [day_index]   (works for the dueling models, e.g. results/full_s0.pt)
import sys
import torch
from env import MicrogridEnv, CAP
from dqn import QNet
from baselines import rule_policy

ACTIONS = ["charge", "discharge", "idle", "charge+shed", "discharge+shed", "idle+shed"]
path = sys.argv[1] if len(sys.argv) > 1 else "results/full_s0.pt"
day = int(sys.argv[2]) if len(sys.argv) > 2 else 150

q = QNet(11, 6, dueling=True)
q.load_state_dict(torch.load(path))
q.eval()

def agent(obs):
    with torch.no_grad():
        return int(q(torch.from_numpy(obs).unsqueeze(0)).argmax())

env = MicrogridEnv("data/solar_hourly.csv", years=[2023], seed=0)
for name, policy in [("trained agent", agent), ("rule baseline", rule_policy)]:
    obs, done = env.reset(day=day, outage=(17, 3)), False
    cost, unserved = 0.0, 0.0
    print(f"--- {name}: test day {day}, outage 17:00-20:00 ---")
    while not done:
        h = env.h
        a = policy(obs)
        obs, r, done, info = env.step(a)
        cost += info["cost"]
        unserved += info["un_crit"]
        print(f"{h:02d}:00  {ACTIONS[a]:15s} battery {env.soc / CAP:4.0%}")
    print(f"total cost Rs {cost:.0f}, unserved critical load {unserved:.1f} kWh\n")