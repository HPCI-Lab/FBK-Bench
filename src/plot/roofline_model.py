
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def get_param(data, context, param): 
    if param in data["activity"][f"context:{context}"].keys():
        return eval(data["activity"][f"context:{context}"][param])#["prov-ml:parameter_value"]
    return None

P_peak = 1e9            # 19.5 TFLOP/s
B_peak = 1.0           # 900 GB/s

USECASES = [file for file in os.listdir("prov/") if "compute" in file]

ts = {"AI": [], "Perf": []}
for USECASE in USECASES: 
    PATH = f"prov/{USECASE}/"
    prov_json = [PATH + file for file in os.listdir(PATH) if file.endswith(".json")][0]
    data = json.load(open(prov_json, "r"))
    exp_name = prov_json.split("/")[-1].removesuffix(".json").removeprefix("prov_")
    ai = get_param(data, exp_name, "arithmetic_intensity")['value:']
    perf = get_param(data, exp_name, "performance")["value:"]
    # print(ai, perf)
    ts["AI"].append(ai)
    ts["Perf"].append(perf)

ls = np.array([10] + ts["AI"] + [1e9, 1e10, 1e11, 1e12, 1e13])
P_mem = B_peak * ls
P_roof = np.minimum(P_peak, P_mem)

plt.title("Roofline Model")
sns.scatterplot(ts, x="Perf", y="AI", marker="X", color="black", label="ML Runs")
plt.xlabel("Operational Intensity (FLOPS/byte)")
plt.ylabel("Attainable Performance (FLOP/s)")
plt.hlines(P_peak, 0, 1e10, colors="red", label="Computational Peak")
plt.plot(ls, P_roof, label="Theoretical Peak")
plt.xlim(1, 10**15)
plt.xscale("log")
plt.yscale("log")
plt.legend(loc="lower right")
# plt.xticks(range(len(mss)), mss)
plt.savefig("imgs/roofline.pdf")
plt.show()