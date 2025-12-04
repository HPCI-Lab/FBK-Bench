# FBK Benchmarks

## Summary 

| Benchmark         | Model Type                     | Dominant Resource                  | Why                                              |
| ----------------- | ------------------------------ | ---------------------------------- | ------------------------------------------------ |
| **Compute-bound** | Very deep MLP                  | Compute / FLOPs                    | Large dense layers; small dataset slice          |
| **Memory-bound**  | Very wide CNN + huge batch     | Memory bandwidth + activation size | Large feature maps create massive memory traffic |
| **I/O-bound**     | Small CNN + heavy augmentation | Disk + CPU dataloader              | Slow transforms + many workers + small batches   |

## Use case 1: Compute bound ML training
## Use case 2: Memory bound ML training
## Use case 3: Simple ML training

## Setup

```bash
pip install -r requirements.txt
```

General folder creation

```bash
python src/preprocess/setup.py
```

Create the MNIST I/O dataset by running: 

```bash
python src/preprocess/create_MNIST_ds.py
```