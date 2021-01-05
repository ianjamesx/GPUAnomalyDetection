# GPU Accelerated Graph Based Anomaly Detection

Extension of NSF REU Research experiment, developed under Salisbury University Henson Grant Research Program

Requires C++17, Nvidia CUDA, CUDA enabled GPU

To run either the SUBDUE implementation or the Barycentric implementation.
Run `nvcc subdue.cu` or `nvcc cluster.cu`

The dataset being read can be found in `dataread.h`

To edit the dataset being read, modify the `readfile` system call in `void init_dataset()` in `dataread.h`

Also note that experiments with memory mapping were being conducted for reading in larger datasets

Note that the dataset must be a pair-value format, the default dataset being used is the KDD CUP 1999 dataset
