import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import v2
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import yprov4ml

SMALL = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomRotation(25),
    v2.ToDtype(torch.float32, scale=True),
])

MEDIUM = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomRotation(25),
    transforms.RandomAffine(0, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    v2.ToDtype(torch.float32, scale=True),
])

LARGE = v2.Compose([
    v2.ToPILImage(), 
    v2.RandomRotation(25),
    v2.RandomAffine(0, translate=(0.2, 0.2)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(25),
    v2.RandomPhotometricDistort(p=1),
    v2.RandomResizedCrop((28, 28), antialias=True),
    v2.RandomRotation(25),
    v2.ColorJitter(brightness=0.5, contrast=0.5),
    v2.ToDtype(torch.float32, scale=True),
])

class MNISTLocalDataset(Dataset): 
    def __init__(self, transform):
        self.path = "data/MNIST/npy/"
        self.files =[self.path + f for f in os.listdir(self.path)]
        self.transform = transform

    def __getitem__(self, index):
        X = torch.tensor(np.load(self.files[index]))
        # X = torch.permute(X, (1,2,0))
        y = self.files[index].split("_")[1]
        return self.transform(X), torch.tensor(int(y))
    
    def __len__(self): 
        return len(self.files)

def io_bound_training(tform, device="cuda"):
    device = torch.device(device)

    if tform == "small": tform = SMALL
    if tform == "medium": tform = MEDIUM
    if tform == "large": tform = LARGE
    trainset = MNISTLocalDataset(tform)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

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

def main(tform): 
    yprov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name=f"io_{tform}", 
        provenance_save_dir="prov",
        save_after_n_logs=100,
        collect_all_processes=False, 
        disable_codecarbon=True, 
        metrics_file_type=yprov4ml.MetricsType.CSV,
    )

    io_bound_training(tform=tform, device="cuda")

    yprov4ml.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tform', default="large", choices=["small", "medium", "large"])      # option that takes a value
    args = parser.parse_args()
    main(args.tform)