import numpy as np, torch, json
from env import MicrogridEnv
from dqn import DQNAgent
from baselines import make_scenarios, evaluate

EPISODES, EVAL_EVERY = 3000, 250
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)

env = MicrogridEnv("data/solar_hourly.csv", years=[2018, 2019, 2020, 2021], seed=SEED)
val_env = MicrogridEnv("data/solar_hourly.csv", years=[2022], seed=SEED)
val_sc = make_scenarios(val_env, n=100, seed=7)

agent = DQNAgent(env.obs_dim, env.n_actions)
log = []
for ep in range(1, EPISODES + 1):
    eps = max(0.05, 1.0 - ep / (0.6 * EPISODES))     # explore a lot early, little later
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
        print(ep, round(eps, 2), "val return", round(float(res["return"]), 2),
              "unserved", round(float(res["unserved_crit_kWh"]), 1))

torch.save(agent.q.state_dict(), "results/dqn_vanilla.pt")
json.dump(log, open("results/dqn_vanilla_log.json", "w"))