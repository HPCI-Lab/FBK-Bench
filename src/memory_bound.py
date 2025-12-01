import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

def memory_bound_training(device="cuda"):
    device = torch.device(device)

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Very large batch size → bandwidth & memory pressure
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096, shuffle=True, num_workers=0)

    # Extremely wide CNN → huge feature maps → memory-bound
    model = nn.Sequential(
        nn.Conv2d(1, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()

    for i, (inputs, labels) in enumerate(trainloader):
        if i == 20:  # shorter run; memory-heavy per step
            break
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if device.type == "cuda" else None
    print("Memory-bound training time:", time.time() - start)

memory_bound_training()
