# Outage-Aware Campus Microgrid Control with Deep RL
RL agent (Dueling Double DQN + PER, optional Lagrangian blackout constraint) for hourly
solar-battery dispatch, trained on real NASA POWER solar data (2018-2023).

## Setup
pip install -r requirements.txt
data/solar_hourly.csv is included. To re-download: python src/fetch_data.py (set LAT/LON first)

## Reproduce
python src/test_env.py                 # sanity check
python src/test_baselines.py           # idle / rule / oracle
python src/train.py --name full --double --dueling --per --seed 0
python src/run_ablation.py 0 1 2       # 5 variants x 3 seeds
# constrained runs (PowerShell):
foreach ($s in 0,1,2) { python src/train.py --name constrained --double --dueling --per --constrained --seed $s }
python src/ablation_table.py ; python src/ablation_table.py --best ; python src/make_plots.py

## Layout
src/env.py (simulator), dqn.py (own DQN), baselines.py (idle/rule/oracle),
train.py, compare.py, ablation_table.py, make_plots.py, results/ (trained agents, logs, figures)

## Assumptions
Battery, panel and load parameters, tariff and outages are assumptions (see src/env.py).
Outages are simulated; the oracle is a benchmark only, not the agent.