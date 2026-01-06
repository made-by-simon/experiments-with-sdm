# Experiments With Sparse Distributed Memory

## Background

Sparse Distributed Memory (SDM) is an associative (distributed) memory model inspired by human longterm memory that stores information across distributed hard locations in a high-dimensional (sparse) binary space. When storing a memory, the address vector activates hard locations within a certain Hamming distance threshold, and the memory vector is written to the locations at those activated locations. During retrieval, the same address (or a similiar address) reactivates similar hard locations, and the stored memory is reconstructed by summing and thresholding the location values across activated locations. SDM exhibits unique behaviour allowing new memories to either reinforce prior memories or cause them to be forgotten. Furthermore, SDM has extremely large and robust storage capacity. 

This repository contains a Python module implementing SDM and a Jupyter notebook conducting various experiements.

## Contents

```
experiments-with-sdm-main/  
├── .gitignore
├── README.md
├── experiments.ipynb
└── kanerva_sdm.py
```

## Requirements

- `numpy` 
- `matplotlib`
- `tqdm`