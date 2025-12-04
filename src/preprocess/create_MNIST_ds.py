

from torchvision import transforms
import torchvision
import numpy as np
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

for i, (sample, y) in enumerate(tqdm(trainset)): 
    np.save(f"data/MNIST/npy/sample_{y}_{i}.npy", sample)