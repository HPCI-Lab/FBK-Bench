import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


USECASES = ["ml_0", "ml_1", "ml_3"]
METRICS = ["Loss"]

for METRIC in METRICS: 

    sns.set_style("darkgrid")

    for USECASE in USECASES: 
        PATH = f"prov/{USECASE}/metrics_GR0/{METRIC}_Context.TRAINING_GR0.csv"
        data = pd.read_csv(PATH, sep=",")
        sns.lineplot(data["Context.TRAINING"], alpha=0.5, errorbar=None)

    plt.xlim((0, 1000))
    plt.legend(["256", "1024", "64"], title="Model Size")
    plt.savefig(f"imgs/{USECASE}_{METRIC}.pdf")
    plt.close()
    # plt.show()