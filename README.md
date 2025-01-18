# RBCB - Really Bad CUDA Benchmarking
Warning: This application contains flashing lights that may trigger seizures in individuals with photosensitive epilepsy. Viewer discretion is advised.

Welcome to RBCB - Really Bad CUDA Benchmarking! This program is designed to benchmark multithreaded performance using both CPU and GPU. It performs matrix multiplication tasks and measures how many cells can be processed in a specified period.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Concepts](#concepts)
  - [Cell](#cell)
  - [Cycle](#cycle)
  - [Threads](#threads)
  - [Performance Evaluation](#performance-evaluation)
- [Performance Grading](#performance-grading)
  - [CPU Grading](#cpu-grading)
  - [GPU Grading](#gpu-grading)
- [Reporting Issues](#reporting-issues)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Multithreaded Benchmarking**: Utilizes multiple threads to perform matrix multiplication tasks.
- **CPU and GPU Support**: Allows you to select between CPU and GPU for benchmarking.
- **Customizable Data Size**: Change the data size for benchmarking to suit your needs.
- **Performance Grading**: Grades the performance based on the total cycles processed.
- **Graphical Rendering Test**: Includes a graphical rendering test to measure FPS and other metrics.

## Getting Started

### Prerequisites

- Windows OS
- CUDA Toolkit installed
- Visual Studio 2021 or later

### Installation

Refer to releases for pre-compiled versions. To build from source:

1. Clone the repository:
```
git clone https://github.com/CatchySmile/RBCB.git
```
2. Open the project in Visual Studio 2022.
3. Build the project to ensure all dependencies are correctly configured.

### Usage

1. Run the program.
2. You will be presented with a menu
```
[1] Start Benchmark
[2] Info
[3] Options [Advanced]
```
3. Select an option:
    - **Begin Benchmark**: Choose between CPU and GPU for benchmarking or run a graphical rendering test.
    - **Info**: Display information about the program.
    - **Options**: Change data size, number of threads, or benchmark duration.
    - **Quit**: Exit the program.

## Concepts

### Cell

A cell in RBCB represents a unit of work that consists of matrix multiplication tasks. Specifically, it involves multiplying two matrices and storing the result in a third matrix. The size of the matrices is determined by the `data_size` parameter. Each cell is processed either by the CPU or GPU, depending on the selected mode.

### Cycle

A cycle refers to the complete execution of the matrix multiplication tasks for a given cell. The program measures the number of cycles processed within the specified period to determine the performance of the CPU or GPU.

### Threads

Threads are used to parallelize the matrix multiplication tasks. When using the CPU, the program creates multiple threads to divide the work among them. Each thread is responsible for processing a portion of the matrix multiplication tasks. This parallelization helps in utilizing the full potential of the CPU or GPU and speeds up the computation.

When using the GPU, the program leverages CUDA to execute the matrix multiplication tasks in parallel on the GPU cores. The CUDA kernel is launched with a grid of threads, where each thread computes a single element of the resulting matrix.

### Performance Evaluation

The performance of the CPU or GPU is evaluated based on the number of cells processed within the specified period. The program calculates the total number of cycles (cells) processed and uses this information to grade the performance. The grading criteria are based on predefined thresholds for the number of cycles processed.

## Performance Grading

The performance of the CPU and GPU is graded based on the total number of cycles processed. The grading criteria are as follows:

### CPU Grading

- **SSS**: ≥ 25,000,000 cycles
- **SS**: ≥ 15,000,000 cycles
- **S**: ≥ 11,500,000 cycles
- **A+**: ≥ 8,500,000 cycles
- **A**: ≥ 6,000,000 cycles
- **B+**: ≥ 4,500,000 cycles
- **B**: ≥ 3,500,000 cycles
- **C+**: ≥ 2,250,000 cycles
- **C**: ≥ 1,500,000 cycles
- **D+**: ≥ 1,300,000 cycles
- **D**: ≥ 900,000 cycles
- **D-**: ≥ 500,010 cycles
- **F**: < 500,010 cycles

### GPU Grading

- **SSS**: ≥ 25,000,000 cycles
- **SS**: ≥ 13,000,000 cycles
- **S**: ≥ 7,500,000 cycles
- **A+**: ≥ 5,000,000 cycles
- **A**: ≥ 4,000,000 cycles
- **B+**: ≥ 3,000,000 cycles
- **B**: ≥ 2,000,000 cycles
- **C+**: ≥ 1,700,000 cycles
- **C**: ≥ 1,500,000 cycles
- **D+**: ≥ 1,100,000 cycles
- **D**: ≥ 900,000 cycles
- **D-**: ≥ 600,001 cycles
- **F**: < 600,001 cycles

## Reporting Issues

If you encounter any errors or issues, please report them at [https://github.com/CatchySmile/RBCB](https://github.com/CatchySmile/RBCB).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by the need to test multithreaded performance and CPU/GPU throughput.
- Special thanks to the contributors and the open-source community.

---

Happy benchmarking!
