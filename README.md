# Models-From-Scratch  

## Overview  
**Models-From-Scratch** is a collection of machine learning models implemented from the ground up in **Python, C, and C++/CUDA**. This project serves as an exploration of machine learning fundamentals while emphasizing **good software engineering practices, testing, and efficient implementation** across multiple languages. My goal is to step one layer of abstraction below libraries like PyTorch and TensorFlow, so that I gain insight into the backends of these frameworks.

By coding these models from scratch, this project aims to:  
- Gain deeper insights into the mathematical foundations of machine learning.  
- Explore the efficiency trade-offs between Python, C, and CUDA implementations.  
- Implement and benchmark optimized machine learning models, including GPU-accelerated versions.  
- Provide a structured, well-tested, and easy-to-use library with FastAPI integration.  

## Implemented Models  
This repository contains (or will contain) implementations of the following models:  

### Classical Models

#### Supervised Learning
- **[Linear Regression](src/classical/linear_regression/README.md)**
- **[Logistic Regression](src/classical/logistic_regression/README.md)**
- **[K-Nearest Neighbors](src/classical/k_nearest_neighbors/README.md)**
- **[Naive Bayes](src/classical/naive_bayes/README.md)**
- **[Decision Tree](src/classical/decision_tree/README.md)**
- **[Random Forest](src/classical/random_forest/README.md)**
- **[Support Vector Machine](src/classical/support_vector_machine/README.md)**

#### Unsupervised Learning
- **[K-Means Clustering](src/classical/k_means/README.md)**
- **[Principal Component Analysis](src/classical/principal_component_analysis/README.md)**

### Deep Learning Models
- **[Feedforward Neural Networks](src/neural/feed_forward/README.md)**
- **[Convolutional Neural Networks](src/neural/convolutional/README.md)**
- **[Graph Neural Networks](src/neural/graph/README.md)**
- **[Recurrent Neural Networks](src/neural/recurrent/README.md)**
- **[Long Short-Term Memory](src/neural/lstm/README.md)**
- **[Transformers](src/neural/transformer/README.md)**

(*Future additions may include generative models.*)  

Each model has its own folder, containing implementations in **Python, C, and CUDA**, along with a dedicated README explaining the approach and optimizations.

## Repository Structure  

The repository is organized as follows:  

```bash
models-from-scratch/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── api
│   └── main.py
├── requirements.txt
├── setup.py
├── src
│   ├── classical
│   │   ├── classical_model.py
│   │   ├── decision_tree
│   │   │   ├── README.md
│   │   │   └── decision_tree.py
│   │   ├── k_means
│   │   │   ├── README.md
│   │   │   └── k_means.py
│   │   ├── k_nearest_neighbors
│   │   │   ├── README.md
│   │   │   └── k_nearest_neighbors.py
│   │   ├── linear_regression
│   │   │   ├── README.md
│   │   │   ├── c_backend
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── linear_regression.c
│   │   │   │   └── linear_regression.h
│   │   │   ├── cuda_backend
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── linear_regression.cpp
│   │   │   │   ├── linear_regression.cu
│   │   │   │   └── linear_regression.h
│   │   │   └── py_interface
│   │   │       ├── linear_regression_base.py
│   │   │       ├── linear_regression_c.py
│   │   │       ├── linear_regression_cuda.py
│   │   │       └── linear_regression_python.py
│   │   ├── logistic_regression
│   │   │   ├── README.md
│   │   │   └── logistic_regression.py
│   │   ├── naive_bayes
│   │   │   ├── README.md
│   │   │   └── naive_bayes.py
│   │   ├── principal_component_analysis
│   │   │   ├── README.md
│   │   │   └── principal_component_analysis.py
│   │   ├── random_forest
│   │   │   ├── README.md
│   │   │   └── random_forest.py
│   │   └── support_vector_machine
│   │       ├── README.md
│   │       └── support_vector_machine.py
│   ├── neural
│   │   ├── convolutional
│   │   │   ├── README.md
│   │   │   └── convolutional.py
│   │   ├── feed_forward
│   │   │   ├── README.md
│   │   │   └── feed_forward.py
│   │   ├── graph
│   │   │   ├── README.md
│   │   │   └── graph.py
│   │   ├── lstm
│   │   │   ├── README.md
│   │   │   └── lstm.py
│   │   ├── recurrent
│   │   │   ├── README.md
│   │   │   └── recurrent.py
│   │   └── transformer
│   │       ├── README.md
│   │       └── transformer.py
│   └── utils
│       ├── c_utils
│       │   ├── CMakeLists.txt
│       │   ├── c_loss_functions
│       │   │   ├── loss_functions.c
│       │   │   └── loss_functions.h
│       │   ├── c_mathematical_functions
│       │   │   ├── pseudoinverse.c
│       │   │   └── pseudoinverse.h
│       │   ├── c_matrix_functions
│       │   │   ├── matrix_functions.c
│       │   │   └── matrix_functions.h
│       │   └── c_memory_functions
│       │       ├── memory_functions.c
│       │       └── memory_functions.h
│       ├── cuda_utils
│       │   ├── CMakeLists.txt
│       │   ├── cuda_loss_functions
│       │   │   ├── loss_functions.cpp
│       │   │   ├── loss_functions.cu
│       │   │   └── loss_functions.h
│       │   ├── cuda_mathematical_functions
│       │   │   ├── pseudoinverse.cu
│       │   │   └── pseudoinverse.h
│       │   ├── cuda_matrix_functions
│       │   │   ├── matrix_functions.cu
│       │   │   └── matrix_functions.h
│       │   └── cuda_memory_functions
│       │       ├── memory_functions.cu
│       │       └── memory_functions.h
│       └── py_utils
└── tests
    ├── classical
    │   └── test_linear_regression.py
    └── neural
```

## Installation  
This project is designed to be packaged as a Python module with C/CUDA extensions. A full API will be available via **FastAPI** with containerized deployment using **Docker and Kubernetes**.  

### Dependencies  
- Python: `numpy`, `sklearn`, `torch` (for benchmarking)  
- C/C++: `gcc`, `g++`, `CMake`  
- CUDA (if using GPU-accelerated models): `CUDA Toolkit`, `nvcc` 

Installation instructions will be available once packaging is finalized.  

## Benchmarks  
Each model includes:  
- **Speed benchmarks** comparing Python, C, and CUDA implementations.  
- **Memory usage analysis** to assess efficiency.  

## Contribution  
If you’d like to contribute, feel free to open issues or pull requests! Guidelines will be provided soon.  
