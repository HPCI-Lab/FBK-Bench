import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import yprov4ml

BATCH_SIZE = 64
PARAMS = 2**6
EPOCHS = 2

def ml_training(device="cuda"):
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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS): 
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            yprov4ml.log_system_metrics(yprov4ml.Context.TRAINING, step=epoch)
            yprov4ml.log_metric("Loss", loss.item(), yprov4ml.Context.TRAINING, step=epoch)
            loss.backward()
            optimizer.step()

# ---------------- yPROV RUN ------------------
yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name=f"ml", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=False, 
    disable_codecarbon=True, 
    metrics_file_type=yprov4ml.MetricsType.CSV,
)

ml_training(device="mps")

yprov4ml.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)
# ---------------------------------------------
