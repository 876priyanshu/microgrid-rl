import argparse, copy, json
import numpy as np, torch
from env import MicrogridEnv, R_SCALE
from dqn import DQNAgent
from baselines import make_scenarios, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="vanilla")
ap.add_argument("--double", action="store_true")
ap.add_argument("--dueling", action="store_true")
ap.add_argument("--per", action="store_true")
ap.add_argument("--constrained", action="store_true")   # learn lambda from a blackout budget
ap.add_argument("--budget", type=float, default=0.3)    # allowed unserved critical kWh per day
ap.add_argument("--eta", type=float, default=0.1)       # how fast lambda reacts
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=3000)
ap.add_argument("--lam_max", type=float, default=20.0)
args = ap.parse_args()

np.random.seed(args.seed); torch.manual_seed(args.seed)
EVAL_EVERY, LAM_EVERY = 250, 20
env = MicrogridEnv("data/solar_hourly.csv", years=[2018, 2019, 2020, 2021], seed=args.seed)
val_env = MicrogridEnv("data/solar_hourly.csv", years=[2022], seed=0)
val_sc = make_scenarios(val_env, n=100, seed=7)

agent = DQNAgent(env.obs_dim, env.n_actions, double=args.double,
                 dueling=args.dueling, per=args.per)
log, recent_u, lam = [], [], 1.0
lam = float(np.clip(lam + args.eta * (np.mean(recent_u) - args.budget), 0.0, args.lam_max))
best_score, best_state = -1e9, None
for ep in range(1, args.episodes + 1):
    eps = max(0.05, 1.0 - ep / (0.6 * args.episodes))
    agent.beta = min(1.0, 0.4 + 0.6 * ep / args.episodes)
    obs, done, ret, u_ep = env.reset(), False, 0.0, 0.0
    while not done:
        a = agent.act(obs, eps)
        obs2, r, done, info = env.step(a)
        r_base = -info["cost"] / R_SCALE                  # reward WITHOUT the critical-load term
        agent.buf.add(obs, a, r_base, obs2, float(done), info["un_crit"])
        agent.learn()
        obs, ret, u_ep = obs2, ret + r, u_ep + info["un_crit"]

    if args.constrained:                                   # dual ascent on lambda
        recent_u.append(u_ep)
        if ep % LAM_EVERY == 0:
            lam = float(np.clip(lam + args.eta * (np.mean(recent_u) - args.budget), 0.0, 20.0))
            agent.lam, recent_u = lam, []

    if ep % EVAL_EVERY == 0:
        res = evaluate(val_env, agent.greedy_policy(), val_sc)
        ret_v, unc_v = float(res["return"]), float(res["unserved_crit_kWh"])
        score = ret_v - 0.5 * max(0.0, unc_v - args.budget)   # prefer checkpoints that respect the budget
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(agent.q.state_dict())
        log.append({"episode": ep, "lambda": lam, "train_return": ret,
                    "return": ret_v, "unserved_crit_kWh": unc_v, "cost_Rs": float(res["cost_Rs"])})
        print(args.name, args.seed, ep, "val return", round(ret_v, 2),
              "unserved", round(unc_v, 1), "lambda", round(lam, 2))

tag = f"results/{args.name}_s{args.seed}"
torch.save(agent.q.state_dict(), tag + ".pt")              # final model
torch.save(best_state, tag + "_best.pt")                   # best validation checkpoint
json.dump(log, open(tag + ".json", "w"))
print("saved", tag)