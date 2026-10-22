import argparse, json
import numpy as np, torch
from env import MicrogridEnv
from dqn import DQNAgent
from baselines import make_scenarios, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="vanilla")
ap.add_argument("--double", action="store_true")
ap.add_argument("--dueling", action="store_true")
ap.add_argument("--per", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=3000)
args = ap.parse_args()

np.random.seed(args.seed); torch.manual_seed(args.seed)
EVAL_EVERY = 250
env = MicrogridEnv("data/solar_hourly.csv", years=[2018, 2019, 2020, 2021], seed=args.seed)
val_env = MicrogridEnv("data/solar_hourly.csv", years=[2022], seed=0)
val_sc = make_scenarios(val_env, n=100, seed=7)

agent = DQNAgent(env.obs_dim, env.n_actions, double=args.double,
                 dueling=args.dueling, per=args.per)
log = []
for ep in range(1, args.episodes + 1):
    eps = max(0.05, 1.0 - ep / (0.6 * args.episodes))
    agent.beta = min(1.0, 0.4 + 0.6 * ep / args.episodes)
    obs, done, ret = env.reset(), False, 0.0
    while not done:
        a = agent.act(obs, eps)
        obs2, r, done, info = env.step(a)
        agent.buf.add(obs, a, r, obs2, float(done))
        agent.learn()
        obs, ret = obs2, ret + r
    if ep % EVAL_EVERY == 0:
        res = evaluate(val_env, agent.greedy_policy(), val_sc)
        log.append({"episode": ep, "train_return": ret, **{k: float(v) for k, v in res.items()}})
        print(args.name, args.seed, ep, "val return", round(float(res["return"]), 2),
              "unserved", round(float(res["unserved_crit_kWh"]), 1))

tag = f"results/{args.name}_s{args.seed}"
torch.save(agent.q.state_dict(), tag + ".pt")
json.dump(log, open(tag + ".json", "w"))
print("saved", tag)
