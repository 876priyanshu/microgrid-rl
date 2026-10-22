import copy
import numpy as np
from env import CAP

# obs: 0 soc | 1,2 hour sin/cos | 3 solar | 4-6 forecast | 7 price/10 | 8 load/300 | 9 outage | 10 warning

def idle_policy(obs):
    return 2                                           # never touch the battery

def rule_policy(obs, reserve=0.3):
    soc, solar, price, load, out, warn = obs[0], obs[3], obs[7]*10, obs[8], obs[9], obs[10]
    if out:                                            # blackout: use battery, shed if low
        return (1 if soc > 0.02 else 2) + (3 if soc < 0.3 else 0)
    if price >= 10:                                    # peak price: use battery down to reserve
        return 1 if soc > reserve else 2
    if solar > load and soc < 0.95:                    # spare sunshine: store it
        return 0
    if (price <= 5 or warn > 0.5) and soc < 0.9:       # cheap power or warning: fill up
        return 0
    return 2

def make_scenarios(env, n, seed=123):
    rng = np.random.default_rng(seed)
    days = rng.integers(env.n_days, size=n)
    sc = []
    for d in days:
        out = (int(rng.integers(8, 21)), int(rng.integers(2, 5))) if rng.random() < env.outage_prob else (-1, 0)
        sc.append((int(d), out))
    return sc

def evaluate(env, policy, scenarios):
    R, C, U = [], [], []
    for day, out in scenarios:
        obs, done, ret, cost, unc = env.reset(day=day, outage=out), False, 0.0, 0.0, 0.0
        while not done:
            obs, r, done, info = env.step(policy(obs))
            ret += r; cost += info["cost"]; unc += info["un_crit"]
        R.append(ret); C.append(cost); U.append(unc)
    return {"return": np.mean(R), "cost_Rs": np.mean(C), "unserved_crit_kWh": np.mean(U)}

def oracle_table(env, n_levels=41):
    """Perfect-foresight dynamic programming for the day loaded in env (BENCHMARK ONLY)."""
    model = copy.copy(env)
    model.rng = np.random.default_rng(0)               # own RNG so env's stream is untouched
    step = CAP / (n_levels - 1)
    V = np.zeros((25, n_levels))
    best = np.zeros((24, n_levels), dtype=int)
    for h in range(23, -1, -1):
        for i in range(n_levels):
            q = []
            for a in range(env.n_actions):
                model.h, model.soc = h, i * step
                _, r, _, _ = model.step(a)
                j = int(round(model.soc / step))
                q.append(r + V[h + 1, j])
            best[h, i] = int(np.argmax(q)); V[h, i] = max(q)
    return best, step

def evaluate_oracle(env, scenarios, n_levels=41):
    R, C, U = [], [], []
    for day, out in scenarios:
        obs, done, ret, cost, unc = env.reset(day=day, outage=out), False, 0.0, 0.0, 0.0
        best, step = oracle_table(env, n_levels)
        while not done:
            a = best[env.h, int(round(env.soc / step))]
            obs, r, done, info = env.step(a)
            ret += r; cost += info["cost"]; unc += info["un_crit"]
        R.append(ret); C.append(cost); U.append(unc)
    return {"return": np.mean(R), "cost_Rs": np.mean(C), "unserved_crit_kWh": np.mean(U)}