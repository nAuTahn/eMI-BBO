# Toward Efficient Mixed-Integer Black-Box Optimization via Evolution Strategies with Plateau Handling Techniques
This repository contains the official code for our paper, *"Toward Efficient Mixed-Integer Black-Box Optimization via Evolution Strategies with Plateau Handling Techniques"* accepted at GECCO 2025.

![overview](fig/high_dimension.png)

## Setup

1. **Clone the repository and navigate to the root directory.**
    ```bash
    git clone https://github.com/nAuTahn/eMI-BBO
    cd eMI-BBO
    ```
2. **Make sure to install the required libraries. Besides [NumPy](https://numpy.org/), you also need [SciPy](https://scipy.org/) to perform some calculations.**

## **Performance evaluation on standard MIP benchmark functions.**
You can reproduce the results in our paper by running the following scripts. Hyperparameters can be adjusted based on the papers or customized by the user.
```bash
python test.py --func ellipsoid_int --dim 120  --dim_co 80 --max_evals 100000 --target 1e-10 --sigma_VD 0.5 --sigma_NES 0.5 --step_size_control "TPA"
```
Another argument `--list_funcs` is added to support displaying the names of some available benchmark functions. Additionally, users can add other objective functions in `test.py` for evaluation.

## Citation

If you use our source code, please cite our work as:

```bibtex
@inproceedings{AnhGecco2025,
  author       = {Tuan Anh Nguyen and Ngoc Hoang Luong},
  title        = {{Toward Efficient Mixed-Integer Black-Box Optimization via Evolution Strategies with Plateau Handling Techniques}},
  booktitle    = {GECCO '25: Proceedings of the Genetic and Evolutionary Computation Conference},
  address      = {Málaga, Spain},
  publisher    = {{ACM}},
  year         = {2025}
}
```

## Acknowledgement
Our source code is inspired by:
- [Comparison-Based Natural Gradient Optimization in High Dimension](https://github.com/akimotolab/RCMA/blob/main/code/vdcma.py)
- [Fast Moving Natural Evolution Strategy for High-Dimensional Problems](https://github.com/nomuramasahir0/crfmnes)
- [Natural Evolution Strategy for Mixed-Integer Black-Box Optimization](https://github.com/ono-lab/dxnesici)
