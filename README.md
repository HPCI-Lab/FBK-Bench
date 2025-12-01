Summary of the Benchmarks

| Benchmark         | Model Type                     | Dominant Resource                  | Why                                              |
| ----------------- | ------------------------------ | ---------------------------------- | ------------------------------------------------ |
| **Compute-bound** | Very deep MLP                  | Compute / FLOPs                    | Large dense layers; small dataset slice          |
| **Memory-bound**  | Very wide CNN + huge batch     | Memory bandwidth + activation size | Large feature maps create massive memory traffic |
| **I/O-bound**     | Small CNN + heavy augmentation | Disk + CPU dataloader              | Slow transforms + many workers + small batches   |
