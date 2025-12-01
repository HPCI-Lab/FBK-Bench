import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

def io_bound_training(device="cuda"):
    device = torch.device(device)

    # Expensive transforms (slow on CPU)
    transform = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor(),
    ])

    # Many workers + small batch = I/O dominated
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
    )

    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()

    # Runtime dominated by DataLoader, not GPU
    for i, (inputs, labels) in enumerate(trainloader):
        if i == 200:  # many batches → more I/O pressure
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    print("I/O-bound training time:", time.time() - start)

io_bound_training()
