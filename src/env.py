import numpy as np
import pandas as pd

# ---- physical constants (our assumptions: state them in the report) ----
CAP     = 400.0    # battery capacity, kWh
P_MAX   = 100.0    # max charge / discharge power, kW
ETA     = 0.95     # one-way efficiency
PV_COEF = 0.32     # kW of solar per W/m^2 of sunlight
CRIT    = 60.0     # critical load, kW
SHAPE = np.array([0.5]*6 + [0.8]*3 + [1.0]*9 + [0.9]*4 + [0.6]*2)
PRICE = np.array([5.0]*6 + [7.0]*12 + [10.0]*4 + [5.0]*2)   # Rs/kWh
C_SHED, C_FORCED, C_CRIT = 15.0, 40.0, 200.0                # Rs/kWh penalties
R_SCALE = 1000.0

class MicrogridEnv:
    """One episode = one day = 24 hourly steps."""
    n_actions = 6   # {charge, discharge, idle} x {keep, shed non-critical}
    obs_dim = 11

    def __init__(self, csv_path, years, outage_prob=0.3, seed=0):
        df = pd.read_csv(csv_path, index_col=0)
        df = df[np.isin(df.index.astype(int) // 1000000, years)]
        n = len(df) // 24
        self.sun = df["ALLSKY_SFC_SW_DWN"].values[:n*24].reshape(n, 24)
        self.temp = df["T2M"].values[:n*24].reshape(n, 24)
        self.n_days = n
        self.outage_prob = outage_prob
        self.rng = np.random.default_rng(seed)
        self.lam = 1.0   # constraint multiplier, used later

    def reset(self, day=None, outage=None):
        self.day = int(self.rng.integers(self.n_days)) if day is None else day
        self.h, self.soc = 0, 0.5 * CAP
        self.pv = np.clip(self.sun[self.day], 0, None) * PV_COEF
        self.noncrit = (80 + 8*np.maximum(self.temp[self.day] - 25, 0)) * SHAPE
        self.out_start, self.out_len = -1, 0
        if outage is not None:                      # fixed scenario: (start_hour, length)
            self.out_start, self.out_len = outage
        elif self.rng.random() < self.outage_prob:
            self.out_start = int(self.rng.integers(8, 21))
            self.out_len = int(self.rng.integers(2, 5))
        return self._obs()
        
            

    def _in_outage(self, h):
        return self.out_start >= 0 and self.out_start <= h < self.out_start + self.out_len

    def _obs(self):
        h = min(self.h, 23)
        fut = np.array([self.pv[min(h+k, 23)] for k in (1, 2, 3)])
        fut = fut * (1 + 0.1 * self.rng.standard_normal(3))       # noisy forecast
        upcoming = any(self._in_outage(h+k) for k in (1, 2, 3))
        warn = float(self.rng.random() < (0.7 if upcoming else 0.1))  # noisy warning
        return np.array([self.soc/CAP, np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                         self.pv[h]/300, *(fut/300), PRICE[h]/10,
                         (CRIT + self.noncrit[h])/300,
                         float(self._in_outage(h)), warn], dtype=np.float32)

    def step(self, a):
        h = self.h
        mode, shed = a % 3, a // 3        # mode: 0 charge, 1 discharge, 2 idle
        out = self._in_outage(h)
        solar = self.pv[h]
        noncrit = 0.0 if shed else self.noncrit[h]
        demand = CRIT + noncrit

        if mode == 0:   p = min(P_MAX, (CAP - self.soc) / ETA)   # p > 0 charging
        elif mode == 1: p = -min(P_MAX, self.soc * ETA)          # p < 0 discharging
        else:           p = 0.0
        if out and p > 0:                  # no grid: charge from spare solar only
            p = min(p, max(solar - demand, 0.0))
        self.soc += p * ETA if p > 0 else p / ETA
        self.soc = float(np.clip(self.soc, 0.0, CAP))

        gap = demand + max(p, 0.0) - solar - max(-p, 0.0)   # > 0: need grid power
        grid = 0.0 if out else max(gap, 0.0)
        unserved = max(gap, 0.0) if out else 0.0
        un_nc = min(unserved, noncrit)       # non-critical is cut first
        un_crit = unserved - un_nc

        shed_kwh = self.noncrit[h] if shed else 0.0
        cost = PRICE[h]*grid + C_SHED*shed_kwh + C_FORCED*un_nc + 0.05*abs(p)
        reward = -(cost + self.lam * C_CRIT * un_crit) / R_SCALE

        self.h += 1
        info = {"cost": cost, "grid": grid, "un_crit": un_crit, "un_nc": un_nc}
        return self._obs(), reward, self.h >= 24, info