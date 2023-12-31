# Performance of AF2 and ESMFold as disorder predictors for de novo proteins and random sequences

This repository contains the code and Data to reproduce the figures published in Middendorf & Eicholt (2023), "Random, de novo and conserved proteins: How
structure and disorder predictors perform differently".

### Content

- `Utils` contains a selection of scripts used in this study.
- `Data` contains all datasets used in this study. A detailed description of the datasets features can be found in the `README.md` file in the `Data` folder.
- `structuralComparisons.ipynb` contains the code to reproduce Figure 1 and Figure S1B
- `correlationsOfFeatures.ipynb` contains the code to reproduce Figure 2, Figure S1A, and Figure S2A, S2B, S2D, S2E
- `per_residue_analysis.ipynb` contains the code to reproduce Figure 3, Figure 4, Figure 5A, Figure S3
- `aa_frquency.ipynb` contains the code to reproduce Figure S4
- `deNovoAgeinfluence.ipynb` contains the code to reproduce Figure 5B&C and Figure S5

### Installation

**Requirements:**
- [anaconda](https://www.anaconda.com/products/individual)

**Usage:**
1. Clone this repository
2. Create a conda environment with the required packages: `conda env create -f environment.yml`
3. Activate the environment: `conda activate Middendorf_Eicholt_2023`
4. Start jupyter notebook from the terminal: `jupyter notebook`
5. Open the notebooks and run the cells

