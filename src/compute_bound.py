import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import argparse
import yprov4ml

class LargeMNISTCNN(nn.Module):
    def __init__(self, width=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),

            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(),

            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((width * 2) * 7 * 7, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

MEM = 0

def compute_bound_training(BATCH_SIZE, PARAMS, device="cuda"):
    global MEM
    MEM = 0
    MODEL_FLOPS = 2 * (28*28 * PARAMS) * PARAMS**2 * 2 * PARAMS * 10
    device = torch.device(device)

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = LargeMNISTCNN(width=PARAMS).to(device)

    def hook_fn(m, inp, out):
        global MEM
        I = [i.numel() for i in inp][0]
        O = out.numel()
        MEM += (I + O) * 4

    for module in model.modules():
        module.register_forward_hook(hook_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()

    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        yprov4ml.log_system_metrics(yprov4ml.Context.TRAINING, step=0)
        loss.backward()
        optimizer.step()

    
    TIME = time.time() - start

    yprov4ml.log_param("arithmetic_intensity", (MODEL_FLOPS / TIME) / MEM)
    yprov4ml.log_param("performance", MODEL_FLOPS / MEM)
    yprov4ml.log_param("batch_size", BATCH_SIZE)
    yprov4ml.log_param("model_params", PARAMS)
    yprov4ml.log_param("memory", MEM)
    yprov4ml.log_param("time", TIME)



def main(BATCH_SIZE, PARAMS): 
    yprov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name=f"compute_b{BATCH_SIZE}_p{PARAMS}", 
        provenance_save_dir="prov",
        save_after_n_logs=100,
        collect_all_processes=False, 
        disable_codecarbon=True, 
        metrics_file_type=yprov4ml.MetricsType.CSV,
    )

    compute_bound_training(BATCH_SIZE, PARAMS, device="cuda")

    yprov4ml.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=1024, type=int, choices=[32, 64, 128, 256, 512, 1024]) 
    parser.add_argument('-p', '--params', default=2**12, type=int, choices=[2**6, 2**8, 2**10, 2**12, 2**14]) 
    args = parser.parse_args()
    main(args.batch_size, args.params)
