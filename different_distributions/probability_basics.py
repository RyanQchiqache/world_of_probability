import math
import numpy as np
import matplotlib.pyplot as plt

# 1. Probability Basics

def conditional_probability(P_A_and_B, P_B):
    """
    Calculate conditional probability P(A|B).

    Args:
        P_A_and_B (float): Probability of A and B.
        P_B (float): Probability of B.

    Returns:
        float: Conditional probability P(A|B).
    """
    return P_A_and_B / P_B

def bayes_theorem(prior_A, likelihood_B_given_A, prior_not_A, likelihood_B_given_not_A):
    """
    Computes the posterior probability of event A given event B.

    Args:
        prior_A (float): The prior probability of event A.
        likelihood_B_given_A (float): The likelihood of event B given that A is true.
        prior_not_A (float): The prior probability of event not A.
        likelihood_B_given_not_A (float): The likelihood of event B given that A is false.

    Returns:
        float: The posterior probability of event A given event B.
    """
    numerator = likelihood_B_given_A * prior_A
    denominator = (likelihood_B_given_A * prior_A) + (likelihood_B_given_not_A * prior_not_A)
    return numerator / denominator

# 2. Discrete Distributions

def bernoulli_distribution(p):
    """
    Generate a Bernoulli distribution.

    Args:
        p (float): Probability of success.

    Returns:
        function: Bernoulli PMF function.
    """
    def pmf(k):
        if k == 1:
            return p
        elif k == 0:
            return 1 - p
        else:
            return 0
    return pmf

def binomial_distribution(n, p):
    """
    Generate a Binomial distribution.

    Args:
        n (int): Number of trials.
        p (float): Probability of success on each trial.

    Returns:
        function: Binomial PMF function.
    """
    def pmf(k):
        if 0 <= k <= n:
            return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        else:
            return 0
    return pmf

def geometric_distribution(p):
    """
    Generate a Geometric distribution.

    Args:
        p (float): Probability of success on each trial.

    Returns:
        function: Geometric PMF function.
    """
    def pmf(k):
        if k >= 1:
            return (1 - p) ** (k - 1) * p
        else:
            return 0
    return pmf

def negative_binomial_distribution(r, p):
    """
    Generate a Negative Binomial distribution.

    Args:
        r (int): Number of successes.
        p (float): Probability of success on each trial.

    Returns:
        function: Negative Binomial PMF function.
    """
    def pmf(k):
        if k >= r:
            return math.comb(k - 1, r - 1) * (p ** r) * ((1 - p) ** (k - r))
        else:
            return 0
    return pmf

def poisson_distribution(mu):
    """
    Generate a Poisson distribution.

    Args:
        mu (float): Expected number of occurrences.

    Returns:
        function: Poisson PMF function.
    """
    def pmf(k):
        if k >= 0:
            return (mu ** k) * (math.exp(-mu)) / math.factorial(k)
        else:
            return 0
    return pmf

# 3. Continuous Distributions

def uniform_distribution(a, b):
    """
    Generate a Uniform distribution.

    Args:
        a (float): Lower bound.
        b (float): Upper bound.

    Returns:
        function: Uniform PDF function.
    """
    def pdf(x):
        if a <= x <= b:
            return 1 / (b - a)
        else:
            return 0
    return pdf

def exponential_distribution(lam):
    """
    Generate an Exponential distribution.

    Args:
        lam (float): Rate parameter (lambda).

    Returns:
        function: Exponential PDF function.
    """
    def pdf(x):
        if x >= 0:
            return lam * math.exp(-lam * x)
        else:
            return 0
    return pdf

def gamma_distribution(alpha, beta):
    """
    Generate a Gamma distribution.

    Args:
        alpha (float): Shape parameter.
        beta (float): Rate parameter (inverse scale).

    Returns:
        function: Gamma PDF function.
    """
    def pdf(x):
        if x >= 0:
            return (beta ** alpha) * (x ** (alpha - 1)) * (math.exp(-beta * x)) / math.gamma(alpha)
        else:
            return 0
    return pdf

def normal_distribution(mu, sigma):
    """
    Generate a Normal distribution.

    Args:
        mu (float): Mean.
        sigma (float): Standard deviation.

    Returns:
        function: Normal PDF ( Probability Density Function ) function.
    """
    def pdf(x):
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return pdf

def normal_cdf(x, mu, sigma):
    """
    Generate a Normal distribution CDF (Cumulative Distribution Function).

    Args:
        x (float): Value at which to evaluate the CDF.
        mu (float): Mean.
        sigma (float): Standard deviation.

    Returns:
        float: CDF value.
    """
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def beta_distribution(alpha, beta):
    """
    Generate a Beta distribution.

    Args:
        alpha (float): Alpha parameter.
        beta (float): Beta parameter.

    Returns:
        function: Beta PDF ( Probability Density Function ) function.
    """
    def pdf(x):
        if 0 <= x <= 1:
            return (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)) / (math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta))
        else:
            return 0
    return pdf

# Plotting Functions for Distributions

def plot_discrete_distribution(pmf, range_vals, title):
    """
    Plot a discrete probability mass function.

    Args:
        pmf (function): Probability mass function.
        range_vals (list): Range of values.
        title (str): Title of the plot.
    """
    probs = [pmf(k) for k in range_vals]
    plt.bar(range_vals, probs)
    plt.xlabel('k')
    plt.ylabel('P(X=k)')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_continuous_distribution(pdf, cdf, mu, sigma, range_vals, title):
    """
    Plot a continuous probability density function and cumulative distribution function.

    Args:
        pdf (function): Probability density function.
        cdf (function): Cumulative distribution function.
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
        range_vals (numpy.ndarray): Range of values.
        title (str): Title of the plot.
    """
    pdf_probs = [pdf(x) for x in range_vals]
    cdf_probs = [cdf(x, mu, sigma) for x in range_vals]

    plt.figure(figsize=(12, 6))

    # Plot PDF
    plt.subplot(1, 2, 1)
    plt.fill_between(range_vals, pdf_probs, alpha=0.6, color='blue', label='PDF')
    plt.plot(range_vals, pdf_probs, label='PDF', color='blue')
    plt.title(f'{title} PDF')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()

    # Plot CDF
    plt.subplot(1, 2, 2)
    plt.plot(range_vals, cdf_probs, label='CDF', color='red')
    plt.title(f'{title} CDF')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 4. Expectation, Variance, and Covariance

def calc_expected_value(distribution, is_continuous=False, lower_bound=0, upper_bound=10, num_points=1000):
    """
    Calculate the expected value of a distribution.

    Args:
        distribution (function): PMF or PDF function.
        is_continuous (bool): Flag indicating if the distribution is continuous.
        lower_bound (float): Lower bound for continuous distributions.
        upper_bound (float): Upper bound for continuous distributions.
        num_points (int): Number of points for numerical integration.

    Returns:
        float: Expected value.
    """
    if is_continuous:
        x_values = np.linspace(lower_bound, upper_bound, num_points)
        return sum([x * distribution(x) for x in x_values]) * (upper_bound - lower_bound) / num_points
    else:
        return sum([k * distribution(k) for k in range(num_points)])

def mean(values):
    """
    Calculate the mean of a list of values.

    Args:
        values (list): List of numerical values.

    Returns:
        float: Mean of the values.
    """
    return sum(values) / len(values)

def calc_variance(distribution, expected_value, is_continuous=False, lower_bound=0, upper_bound=10, num_points=1000):
    """
    Calculate the variance of a distribution.

    Args:
        distribution (function): PMF or PDF function.
        expected_value (float): Expected value of the distribution.
        is_continuous (bool): Flag indicating if the distribution is continuous.
        lower_bound (float): Lower bound for continuous distributions.
        upper_bound (float): Upper bound for continuous distributions.
        num_points (int): Number of points for numerical integration.

    Returns:
        float: Variance.
    """
    if is_continuous:
        x_values = np.linspace(lower_bound, upper_bound, num_points)
        return sum([(x - expected_value) ** 2 * distribution(x) for x in x_values]) * (upper_bound - lower_bound) / num_points
    else:
        return sum([(k - expected_value) ** 2 * distribution(k) for k in range(num_points)])

def covariance(X, Y):
    """
    Calculate the covariance between two variables.

    Args:
        X (list): First variable.
        Y (list): Second variable.

    Returns:
        float: Covariance.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of elements")

    mean_X = mean(X)
    mean_Y = mean(Y)
    covariance = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(len(X))) / len(X)

    return covariance

def standard_deviation(values):
    """
    Calculate the standard deviation of a list of values.

    Args:
        values (list): List of numerical values.

    Returns:
        float: Standard deviation of the values.
    """
    return math.sqrt(sum([(x - mean(values)) ** 2 for x in values]) / len(values))

def correlation(X, Y):
    """
    Calculate the correlation between two variables.

    Args:
        X (list): First variable.
        Y (list): Second variable.

    Returns:
        float: Correlation coefficient.
    """
    cov = covariance(X, Y)
    std_X = standard_deviation(X)
    std_Y = standard_deviation(Y)

    return cov / (std_X * std_Y)

# 5. Law of Large Numbers

def law_of_large_numbers(n_trials, dist, sample_size=1):
    """
    Demonstrate the Law of Large Numbers.

    Args:
        n_trials (int): Number of trials.
        dist (function): Distribution function.
        sample_size (int): Number of samples per trial.

    Returns:
        numpy.ndarray: Sample means.
    """
    sample_means = [np.mean([dist() for _ in range(sample_size)]) for _ in range(n_trials)]
    return np.array(sample_means)

def plot_law_of_large_numbers(sample_means, true_mean):
    """
    Plot the Law of Large Numbers demonstration.

    Args:
        sample_means (numpy.ndarray): Sample means.
        true_mean (float): True mean of the distribution.
    """
    plt.plot(sample_means, label='Sample Mean')
    plt.axhline(true_mean, color='r', linestyle='--', label='True Mean')
    plt.xlabel('Number of Trials')
    plt.ylabel('Sample Mean')
    plt.title('Law of Large Numbers')
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. Central Limit Theorem

def central_limit_theorem(n_samples, sample_size, dist):
    """
    Demonstrate the Central Limit Theorem.

    Args:
        n_samples (int): Number of samples.
        sample_size (int): Size of each sample.
        dist (function): Distribution function.

    Returns:
        numpy.ndarray: Sample means.
    """
    sample_means = [np.mean([dist() for _ in range(sample_size)]) for _ in range(n_samples)]
    return np.array(sample_means)

def normal_pdf(x, mu, sigma):
    """
    Calculate the PDF of a normal distribution.

    Args:
        x (float): Value at which to evaluate the PDF.
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.

    Returns:
        float: PDF value.
    """
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normal_cdf(x, mu, sigma):
    """
    Calculate the CDF of a normal distribution.

    Args:
        x (float): Value at which to evaluate the CDF.
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.

    Returns:
        float: CDF value.
    """
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def plot_central_limit_theorem(sample_means):
    """
    Plot the Central Limit Theorem demonstration.

    Args:
        sample_means (numpy.ndarray): Sample means.
    """
    plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='g')

    mu = np.mean(sample_means)
    sigma = np.std(sample_means)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = [normal_pdf(xi, mu, sigma) for xi in x]
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Central Limit Theorem')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

# 7. Bayesian Inference

def bayesian_inference(prior, likelihood, data):
    """
    Perform Bayesian inference.

    Args:
        prior (function): Prior distribution function.
        likelihood (function): Likelihood function.
        data (list): Observed data.

    Returns:
        function: Posterior distribution function.
    """
    def posterior(x):
        return prior(x) * np.prod([likelihood(x, datum) for datum in data])
    return posterior

# 8. Markov Chains

def markov_chain(transition_matrix, initial_state, n_steps):
    """
    Simulate a Markov chain.

    Args:
        transition_matrix (numpy.ndarray): Transition matrix.
        initial_state (int): Initial state.
        n_steps (int): Number of steps.

    Returns:
        numpy.ndarray: Sequence of states.
    """
    n_states = transition_matrix.shape[0]
    states = [initial_state]
    for _ in range(n_steps):
        current_state = states[-1]
        next_state = np.random.choice(n_states, p=transition_matrix[current_state])
        states.append(next_state)
    return np.array(states)

# 9. Linear Regression

def linear_regression(X, y):
    """
    Perform linear regression.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.

    Returns:
        numpy.ndarray: Regression coefficients.
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

def plot_regression_line(X, y, coefficients):
    """
    Plot the regression line.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        coefficients (numpy.ndarray): Regression coefficients.
    """
    plt.scatter(X[:, 1], y, color='blue', label='Data Points')
    plt.plot(X[:, 1], X @ coefficients, color='red', label='Regression Line')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to demonstrate examples
def main():
    # Example for Bayes' Theorem
    prior_A = 0.01  # Example: probability of having the disease
    likelihood_B_given_A = 0.9  # Example: probability of a positive test given the disease
    prior_not_A = 0.99  # Example: probability of not having the disease
    likelihood_B_given_not_A = 0.05  # Example: probability of a positive test without the disease

    posterior = bayes_theorem(prior_A, likelihood_B_given_A, prior_not_A, likelihood_B_given_not_A)
    print(f"Posterior Probability (Bayes' Theorem): {posterior:.4f}")

    # Example for Binomial Distribution
    n = 100  # Example: number of trials
    pi = 0.5  # Example: probability of success on a single trial

    binom_dist = binomial_distribution(n, pi)
    exp_val = calc_expected_value(binom_dist)
    var = calc_variance(binom_dist, exp_val)
    print(f"Binomial Distribution - Expected Value: {exp_val}")
    print(f"Binomial Distribution - Variance: {var}")
    plot_discrete_distribution(binom_dist, range(n + 1), "Binomial Distribution")

    # Example for Normal Distribution
    mu = 175  # Mean
    sigma = 10  # Standard deviation

    norm_pdf = normal_distribution(mu, sigma)
    exp_val = calc_expected_value(norm_pdf, is_continuous=True, lower_bound=140, upper_bound=220)
    var = calc_variance(norm_pdf, exp_val, is_continuous=True, lower_bound=140, upper_bound=220)
    print(f"Normal Distribution - Expected Value: {exp_val}")
    print(f"Normal Distribution - Variance: {var}")
    plot_continuous_distribution(norm_pdf, normal_cdf, mu, sigma, np.linspace(140, 220, 1000), "Normal Distribution")

    # Law of Large Numbers Demonstration
    norm_rv = lambda: np.random.normal(mu, sigma)
    sample_means = law_of_large_numbers(10000, norm_rv)
    plot_law_of_large_numbers(sample_means, mu)

    # Central Limit Theorem Demonstration
    sample_means = central_limit_theorem(10000, 1000, norm_rv)
    plot_central_limit_theorem(sample_means)

    # Linear Regression Example
    X = np.array([[1, i] for i in range(10)])  # Feature matrix with intercept term
    y = np.array([2 * i + 1 for i in range(10)])  # Target vector

    coefficients = linear_regression(X, y)
    print(f"Linear Regression Coefficients: {coefficients}")

    plot_regression_line(X, y, coefficients)

# Call the main function
if __name__ == "__main__":
    main()
