import sys
import subprocess

seeds = [
    601,
    143,
    845,
    630,
    459,
    123,
    201,
    221,
    58,
    157,
    414,
    42,
    991,
    676,
    115,
    826,
    399,
    79,
    150,
    711
]

dataset = sys.argv[1]

for seed in seeds:
    print(f"seed: {seed}")
    subprocess.run(
        f"python DeepST_run.py drafts/config_{dataset}.json {seed}", shell=True)
