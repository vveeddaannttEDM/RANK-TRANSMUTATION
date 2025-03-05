import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

def loss_function(params, data):
    """Loss function to estimate lambda by minimizing skew and kurtosis difference."""
    lambda_ = params[0]
    sample_skew = skew(data)
    sample_kurt = kurtosis(data)
    expected_skew = lambda_ * (2 / np.sqrt(np.pi))
    expected_kurt = 3 + (13 / (2 * np.pi)) * lambda_
    return (sample_skew - expected_skew)**2 + (sample_kurt - expected_kurt)**2

def estimate_lambda(data):
    """Estimates the best lambda value for a given dataset."""
    result = minimize(loss_function, x0=[0], args=(data,), bounds=[(-1, 1)])
    return result.x[0]

# Example usage
if __name__ == "__main__":
    sample_data = np.random.normal(0, 1, 1000)
    estimated_lambda = estimate_lambda(sample_data)
    print(f"Estimated lambda: {estimated_lambda}")
