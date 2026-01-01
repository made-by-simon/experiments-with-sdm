# Experiments With Sparse Distributed Memory

## Background

This repository is a collection of code related to experiments with Sparse Distributed Memory, including Python code, Jupyter notebooks, and C++ code. 

Sparse Distributed Memory (SDM) is an associative (distributed) memory model inspired by human longterm memory that stores information across distributed hard locations in a high-dimensional (sparse) binary space. When storing a memory, the address vector activates hard locations within a certain Hamming distance threshold, and the memory vector is written to the locations at those activated locations. During retrieval, the same address (or a similiar address) reactivates similar hard locations, and the stored memory is reconstructed by summing and thresholding the location values across activated locations. SDM exhibits unique behaviour allowing new memories to either reinforce prior memories or cause them to be forgotten. Furthermore, SDM has extremely large and robust storage capacity. 

## Contents

- Readme file `README.md`: Information file (you are currently reading it).
- Jupyter notebook `experiments.ipynb`: Various experiments with Sparse Distributed Memory.
- Python code `kanerva_sdm.py`: Implementation of core SDM functionality (initializing, writing, reading, erasing) in Python. 
- Folder `cpp`: Files and code related to implementing SDM in C++. 
    - C++ header `kanerva_sdm.h`: Implementation of core SDM functionality (initializing, writing, reading, erasing) in C++. 
    - Folder `bindings`: Files and code for creating Python bindings to the C++ SDM implementation. 
    - 
## Requirements


