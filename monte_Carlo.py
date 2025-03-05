import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from rank_transmutation import inverse_quadratic_rtm

def monte_carlo_sampling(lambda_, size=10000):
    """Generates Monte Carlo samples from a transmuted normal distribution."""
    u_samples = np.random.uniform(0, 1, size)
    skewed_samples = stats.norm.ppf(inverse_quadratic_rtm(u_samples, lambda_))
    return skewed_samples

# Example usage
if __name__ == "__main__":
    lambda_ = 0.5
    samples = monte_carlo_sampling(lambda_)
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')
    plt.title("Monte Carlo Sampling of Skewed Normal Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.show()
