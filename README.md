# RANK-TRANSMUTATION
# Rank Transmutation Maps (RTM) Implementation

## Overview
This repository provides an implementation of **Rank Transmutation Maps (RTM)**, as introduced in the paper *"The Alchemy of Probability Distributions: Beyond Gram-Charlier Expansions"* by William T. Shaw and Ian R. C. Buckley. RTMs offer a powerful way to introduce **skewness and kurtosis** into probability distributions while maintaining mathematical tractability.

## Features
- **Quadratic Rank Transmutation Map (QRTM)**
- **Transformation of uniform, exponential, and normal distributions**
- **Monte Carlo sampling of transmuted distributions**
- **Visualizations of skewed distributions**

## Installation
Clone this repository and install dependencies:
```bash
git clone <repo_link>
cd rank-transmutation
pip install -r requirements.txt
```

## Usage
### 1. Skewed Distribution Visualization
Run the Python script to visualize the effect of transmutation on a normal distribution:
```bash
python rank_transmutation.py
```
This will generate plots of skewed normal distributions for different values of the skew parameter **λ**.

### 2. Monte Carlo Sampling
The script also demonstrates how to generate **random samples** from a transmuted distribution using the inverse QRTM method.

## Example Code
```python
from scipy.stats import norm
import numpy as np

def quadratic_rtm(u, lambda_):
    return u + lambda_ * u * (1 - u)

# Generate transmuted samples
u_samples = np.random.uniform(0, 1, 10000)
skewed_samples = norm.ppf(quadratic_rtm(u_samples, lambda_=0.5))
```

## References
- [Original Paper](https://arxiv.org/abs/0901.0434v1)
- Azzalini A., "The Skew-Normal Distribution and Related Families"

## License
MIT License

