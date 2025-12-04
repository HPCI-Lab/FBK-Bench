import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import yprov4ml

BATCH_SIZE = 1024
PARAMS = 2**14
MEM = 0

def compute_bound_training(device="cuda"):
    global MEM
    MEM = 0
    MODEL_FLOPS = 2 * (28*28 * PARAMS) * PARAMS**2 * 2 * PARAMS * 10
    device = torch.device(device)

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, PARAMS),
        nn.ReLU(),
        nn.Linear(PARAMS, PARAMS),
        nn.ReLU(),
        nn.Linear(PARAMS, PARAMS),
        nn.ReLU(),
        nn.Linear(PARAMS, 10)
    ).to(device)

    def hook_fn(m, inp, out):
        global MEM
        I = [i.numel() for i in inp][0]
        O = out.numel()
        MEM += (I + O) * 4

    for module in model.modules():
        module.register_forward_hook(hook_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()

    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        yprov4ml.log_system_metrics(yprov4ml.Context.TRAINING, step=0)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    
    TIME = time.time() - start
    # print("Compute-bound training time:", TIME)

    return {
        "AI": (MODEL_FLOPS / TIME) / MEM,
        "Perf": MODEL_FLOPS / MEM,
        "BS": BATCH_SIZE,
        "MS": PARAMS,
        "MEM": MEM, 
        "TIME": TIME
    }

# ------------ MODEL SIZE CHECK ---------------
# ts = []
# mss = [2**i for i in range(6, 15)]
# for ms in mss: 
#     PARAMS = ms
#     ts.append(compute_bound_training(device="mps"))
# ts = pd.DataFrame(ts)
# ts.to_csv("csvs/MODEL_SIZE_CHECK.csv")
# ts = pd.read_csv("csvs/MODEL_SIZE_CHECK.csv")

# P_peak = 1e9            # 19.5 TFLOP/s
# B_peak = 1.0           # 900 GB/s

# # 3. Compute theoretical roofline
# ls = np.array([10] + ts["AI"].tolist() + [1e9, 1e10, 1e11, 1e12, 1e13])
# P_mem = B_peak * ls
# P_roof = np.minimum(P_peak, P_mem)

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.title("Roofline Model")
# sns.scatterplot(ts, x="Perf", y="AI", marker="X", color="black", label="ML Runs")
# plt.xlabel("Operational Intensity (FLOPS/byte)")
# plt.ylabel("Attainable Performance (FLOP/s)")
# plt.hlines(P_peak, 0, 1e10, colors="red", label="Computational Peak")
# plt.plot(ls, P_roof, label="Theoretical Peak")
# plt.xlim(1, 10**15)
# plt.xscale("log")
# plt.yscale("log")
# plt.legend(loc="lower right")
# # plt.xticks(range(len(mss)), mss)
# plt.savefig("imgs/MODEL_SIZE_CHECK.png")
# plt.show()
# ---------------------------------------------



# ---------------- yPROV RUN ------------------
yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name=f"compute_b{BATCH_SIZE}_p{PARAMS}", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=False, 
    disable_codecarbon=True, 
    metrics_file_type=yprov4ml.MetricsType.CSV,
)

compute_bound_training(device="mps")

yprov4ml.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)
# ---------------------------------------------
