import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

def compute_bound_training(device="cuda"):
    device = torch.device(device)

    # MNIST
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=0)

    # Deep dense network = compute heavy
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()

    for i, (inputs, labels) in enumerate(trainloader):
        if i == 50:  # short run, but very compute heavy
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    print("Compute-bound training time:", time.time() - start)

compute_bound_training(device="mps")
