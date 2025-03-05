# Rank Transmutation Maps (RTM) Implementation
# Based on "The Alchemy of Probability Distributions"

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def quadratic_rtm(u, lambda_):
    """Applies the Quadratic Rank Transmutation Map (QRTM)."""
    return u + lambda_ * u * (1 - u)

def inverse_quadratic_rtm(u, lambda_):
    """Inverse of the QRTM for generating samples."""
    return (1 + lambda_ - np.sqrt((1 + lambda_)**2 - 4 * lambda_ * u)) / (2 * lambda_)

def transmuted_pdf(x, base_cdf, base_pdf, lambda_):
    """Computes the PDF of the transmuted distribution."""
    return base_pdf(x) * (1 + lambda_ - 2 * lambda_ * base_cdf(x))

# Example: Skew-Normal Transformation
x = np.linspace(-3, 3, 1000)
base_cdf = stats.norm.cdf(x)
base_pdf = stats.norm.pdf(x)
lambda_values = [-0.8, -0.4, 0, 0.4, 0.8]

plt.figure(figsize=(8, 6))
for lambda_ in lambda_values:
    plt.plot(x, transmuted_pdf(x, stats.norm.cdf, stats.norm.pdf, lambda_), label=f'lambda={lambda_}')

plt.legend()
plt.title("Skewed Normal Distributions via QRTM")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid()
plt.show()

# Monte Carlo Sampling
u_samples = np.random.uniform(0, 1, 10000)
skewed_samples = stats.norm.ppf(inverse_quadratic_rtm(u_samples, lambda_=0.5))
plt.hist(skewed_samples, bins=50, density=True, alpha=0.6, color='g')
plt.title("Monte Carlo Sampling of Skewed Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()
