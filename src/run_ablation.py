import subprocess, sys

configs = {"vanilla": [], "double": ["--double"], "dueling": ["--dueling"],
           "per": ["--per"], "full": ["--double", "--dueling", "--per"]}
seeds = sys.argv[1:] or ["0"]
for seed in seeds:
    for name, flags in configs.items():
        subprocess.run([sys.executable, "src/train.py", "--name", name,
                        "--seed", seed, *flags], check=True)