import json, glob
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env import MicrogridEnv, CAP, PRICE
from dqn import QNet
from baselines import rule_policy

# ---------- 1. learning curves (validation return, mean +- std over seeds) ----------
def curves(name):
    files = sorted(glob.glob(f"results/{name}_s[0-9].json"))
    if not files:
        return None
    runs = [json.load(open(f)) for f in files]
    x = [r["episode"] for r in runs[0]]
    y = np.array([[r["return"] for r in run] for run in runs])
    return x, y.mean(0), y.std(0)

plt.figure(figsize=(3.4, 2.6))
for name, label in [("vanilla", "DQN"), ("full", "Dueling Double DQN + PER"),
                    ("constrained", "+ blackout constraint")]:
    c = curves(name)
    if c:
        x, m, s = c
        plt.plot(x, m, label=label); plt.fill_between(x, m - s, m + s, alpha=0.2)
plt.xlabel("training episode"); plt.ylabel("validation return")
plt.legend(fontsize=6); plt.tight_layout()
plt.savefig("results/fig_learning.png", dpi=200); plt.close()

# ---------- 2. ablation bars (held-out test year) ----------
tab = json.load(open("results/ablation_test.json"))
try:
    tab_best = json.load(open("results/ablation_test_best.json"))
except FileNotFoundError:
    tab_best = {}
rows = [(n, n, tab[n]) for n in ["vanilla", "double", "dueling", "per", "full", "constrained", "constrained_v2"] if n in tab]
for n in ["constrained", "constrained_v2"]:
    if n in tab_best:
        rows.append((n + "\n(best ckpt)", n, tab_best[n]))
labels = [r[0] for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
ax[0].bar(labels, [r[2]["gap_mean"] for r in rows], yerr=[r[2]["gap_std"] for r in rows], capsize=2)
ax[0].set_ylabel("% of rule-to-oracle gap closed")
ax[1].bar(labels, [r[2]["unserved_mean"] for r in rows], yerr=[r[2]["unserved_std"] for r in rows], capsize=2)
ax[1].axhline(tab["rule"]["unserved_crit_kWh"], ls="--", color="k", label="rule baseline")
ax[1].set_ylabel("unserved critical load (kWh/day)"); ax[1].legend(fontsize=6)
for a in ax:
    a.tick_params(axis="x", labelrotation=60, labelsize=6)
plt.tight_layout(); plt.savefig("results/fig_ablation.png", dpi=200); plt.close()

# ---------- 3. what did the agent learn? (battery level on one test day) ----------
def load(path, dueling):
    q = QNet(11, 6, dueling=dueling)
    q.load_state_dict(torch.load(path)); q.eval()
    def pol(obs):
        with torch.no_grad():
            return int(q(torch.from_numpy(obs).unsqueeze(0)).argmax())
    return pol

def trace(env, policy, day, outage):
    obs, done = env.reset(day=day, outage=outage), False
    soc = [env.soc / CAP]
    while not done:
        obs, r, done, info = env.step(policy(obs))
        soc.append(env.soc / CAP)
    return soc

env = MicrogridEnv("data/solar_hourly.csv", years=[2023], seed=0)
agent = load("results/full_s0.pt", True)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for ax, (title, out) in zip(axes, [("normal day", (-1, 0)), ("outage 17:00-20:00", (17, 3))]):
    ax.plot(trace(env, agent, 150, out), label="DQN agent")
    ax.plot(trace(env, rule_policy, 150, out), "--", label="rule")
    ax.step(range(24), PRICE / PRICE.max(), where="post", color="gray", alpha=0.5, label="price (scaled)")
    if out[0] >= 0:
        ax.axvspan(out[0], out[0] + out[1], color="red", alpha=0.15)
    ax.set_title(title, fontsize=8); ax.set_xlabel("hour of day")
axes[0].set_ylabel("battery level (fraction)"); axes[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig("results/fig_policy.png", dpi=200); plt.close()
print("saved results/fig_learning.png, fig_ablation.png, fig_policy.png")