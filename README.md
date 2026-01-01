# Experiments With Sparse Distributed Memory

## Background

This repository is a collection of code related to experiments with Sparse Distributed Memory, including Python code, Jupyter notebooks, and C++ code. 

Sparse Distributed Memory (SDM) is an associative (distributed) memory model inspired by human longterm memory that stores information across distributed hard locations in a high-dimensional (sparse) binary space. When storing a memory, the address vector activates hard locations within a certain Hamming distance threshold, and the memory vector is written to the locations at those activated locations. During retrieval, the same address (or a similiar address) reactivates similar hard locations, and the stored memory is reconstructed by summing and thresholding the location values across activated locations. SDM exhibits unique behaviour allowing new memories to either reinforce prior memories or cause them to be forgotten. Furthermore, SDM has extremely large and robust storage capacity. 

## Contents

```
experiments-with-sdm-main/  
├── cpp/                       
│   ├── src/                    
│   │   ├── bindings.cpp
│   │   └── kanerva_sdm.h
│   ├── kanerva_sdm.py
│   ├── pyproject.toml
│   ├── setup.py
│   └── speed_test.ipynb
├── .gitignore
├── README.md
├── experiments.ipynb
└── kanerva_sdm.py
```

## Requirements


