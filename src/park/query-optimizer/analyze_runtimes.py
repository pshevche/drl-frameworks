import json
import pdb
import pandas as pd
from collections import defaultdict

file_name = "allQueryRuntimes.json"
# file_name = "allQueryRuntimes-trainedOnCM.json"
# file_name = "allQueryRuntimes-debug.json"
with open(file_name, "r") as f:
    data = json.loads(f.read())
# TODO: worst queries, best queries, avg runtime across all runs etc.

# key: plannerName : [runtimes] (last one or avg)
all_rts = defaultdict(list)

for q, planners in data.items():
    for plannerName, vals in planners.items():
        all_rts[plannerName].append(vals[-1])

for plannerName, vals in all_rts.items():
    avg = sum(vals) / len(vals)
    print("{} avg runtime: {}, num queries: {}".format(plannerName, avg,
        len(vals)))
