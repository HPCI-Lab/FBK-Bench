import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import argparse
import yprov4ml

from model import LargeMNISTCNN

def ml_training(EPOCHS, PARAMS, SAMPLES, device="cuda"):
    BATCH_SIZE = 64
    device = torch.device(device)

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    if SAMPLES: 
        trainset = Subset(trainset, range(SAMPLES))
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LargeMNISTCNN(width=PARAMS).to(device)

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

    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    if SAMPLES: 
        testset = Subset(trainset, range(SAMPLES))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    yprov4ml.log_param("Test Loss", avg_loss)
    yprov4ml.log_param("Test Accuracy", accuracy)
    print(accuracy*100)


def main(EPOCHS, PARAMS, SAMPLES): 
    yprov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name=f"ml_{PARAMS}_{EPOCHS}_{SAMPLES}", 
        provenance_save_dir="prov",
        save_after_n_logs=100,
        collect_all_processes=False, 
        disable_codecarbon=True, 
        metrics_file_type=yprov4ml.MetricsType.CSV,
    )

    ml_training(EPOCHS, PARAMS, SAMPLES, device="cuda")

    yprov4ml.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1, choices=[1, 3, 5, 7, 9]) 
    parser.add_argument('-p', '--params', type=int, default=2**10, choices=[2**6, 2**8, 2**10, 2**12]) 
    parser.add_argument('-s', '--samples', type=int, default=2**12, choices=[2**10, 2**12, 2**14, None]) 
    args = parser.parse_args()
    main(args.epochs, args.params, args.samples)
