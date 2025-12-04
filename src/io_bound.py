import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import numpy as np
from torch.utils.data import Dataset
import os
import yprov4ml

class MNISTLocalDataset(Dataset): 
    def __init__(self):
        self.path = "data/MNIST/npy/"
        self.files =[self.path + f for f in os.listdir(self.path)]
        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.RandomRotation(25),
            transforms.RandomAffine(0, translate=(0.2, 0.2)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        X = np.load(self.files[index])
        X = np.permute_dims(X, axes=(1,2,0))
        y = self.files[index].split("_")[1]
        return self.transform(X), torch.tensor(int(y))
    
    def __len__(self): 
        return len(self.files)

def io_bound_training(device="cuda"):
    device = torch.device(device)

    trainset = MNISTLocalDataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        yprov4ml.log_system_metrics(yprov4ml.Context.TRAINING, step=0)
        loss.backward()
        optimizer.step()

    # yprov4ml.log_carbon_metrics(yprov4ml.Context.TRAINING, step=0)

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name=f"io", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=False, 
    disable_codecarbon=True, 
    metrics_file_type=yprov4ml.MetricsType.CSV,
)

io_bound_training(device="mps")

yprov4ml.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)
