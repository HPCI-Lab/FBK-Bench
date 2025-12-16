import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


USECASE = "compute_b64_p256_1" # "io_15"
METRICS = ["cpu_usage", "gpu_usage", "gpu_memory_usage", "memory_usage"]

for METRIC in METRICS: 
    PATH = f"prov/{USECASE}/metrics_GR0/{METRIC}_Context.TRAINING_GR0.csv"

    data = pd.read_csv(PATH, sep=",")

    sns.set_style("darkgrid")
    sns.lineplot(data["Context.TRAINING"])
    plt.savefig(f"imgs/{USECASE}_{METRIC}.pdf")
    plt.close()
    # plt.show()