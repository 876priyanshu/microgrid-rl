from env import MicrogridEnv

env = MicrogridEnv("data/solar_hourly.csv", years=[2018, 2019, 2020, 2021])
obs, done, total = env.reset(), False, 0
while not done:
    obs, r, done, info = env.step(2)   # always idle
    total += r
print(total)