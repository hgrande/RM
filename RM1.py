import numpy as np
import matplotlib.pyplot as plt
import reprlib
import dimod
import neal
from dwave.system.composites import EmbeddingComposite

np.random.seed(0)
repr_compact = reprlib.Repr()
repr_compact.maxother=200

from scipy import stats
from sklearn.linear_model import LinearRegression
from pyqubo import Binary


a0=[-0.3, -0.5, -1.0, -2.0, -3.0, -3.3, -3.5]  # elasticities in linear demand model
b0=300  # constants in the linear demand model 
sigma=10. # the standard deviation of the noise
price_levels=[5, 8, 10, 12, 13, 16, 19] # predefined possible prices for each day
probabilities=[0.3, 0.3, 0.2, 0.05, 0.05, 0.05, 0.05]  # the probabilities of taking a price choice at a day
n_samples=1000  # the number of sample we want to create

def create_data_point(p, a, b, sigma):
    """
    estimate the demand
    :param p: np.array, (T,)
    :param a: np.array, (T,)
    :param b: float, the constants
    :return: v
    """
    v = np.dot(p,a) + b + np.random.normal(loc=0.0, scale=sigma)
    return v

def create_dataset(a, b, N, price_levels, probabilities, sigma):
    """
    create a dataset for training the demand model
    :param a: np.array, (T,)
    :param b: float
    :param N: int number of samples
    :param price_levels: list, price levels
    :param probabilities: list, probabilities distribution of the prices
    :return:
    """
    t = len(a)
    prices = np.random.choice(price_levels, N+t-1, p=probabilities)
    data_x = []
    data_y = []
    for i in range(N):
        p = prices[i:i+t]
        v = create_data_point(p, a, b, sigma)
        data_x.append(
            np.expand_dims(p, axis=0)
        )
        data_y.append(v)

    data_x = np.concatenate(data_x,axis=0)
    return data_x, data_y

data_x, data_y = create_dataset(a0, b0, n_samples, price_levels, probabilities, sigma)

print("dataset x first 10 samples:")
print(data_x[:10])
print("dataset y first 10 samples:")
print(data_y[:10])

def linear_regression(data_x, data_y):
    reg = LinearRegression().fit(data_x, data_y)
    a = reg.coef_
    b = reg.intercept_
    return a, b


a, b = linear_regression(data_x, data_y)
a = [i for i in a]
b = b

print(f'fitted elasticities: {a}')
print(f'fitted constant: {b}')

p_data = data_x[np.random.randint(0,n_samples)].tolist()

print(p_data)

t = len(a) # next number of days to optimize
n_level = len(price_levels)  # number of price options

x = []
p = []

# get p
for i in range(t):
    p_i = 0
    for j in range(n_level):
        x_ij = Binary(f"X_{i*n_level+j:03d}")
        x.append(x_ij)
        p_i += x_ij*price_levels[j]
    p.append(p_i)
    
# plus historical prices
all_p = p_data + p

print(all_p)

def get_demand(coeff, b, prices):
    assert len(coeff) == len(prices)
    d = b
    for i in range(len(coeff)):
        d += coeff[i]*prices[i]

    return d

# get d, rev
def get_demands_rev(a,b,hist_p, p):
    """ represent the next n days demands and total revenue, based on:
        1. the fitted coefficients a,b
        2. the historical price hist_p
        3. and the future price decisions represented by binary variable x
    """
    all_p = hist_p + p
    t = len(a)
    d = [get_demand(
            coeff=a,
            b=b,
            prices=all_p[i+1:i+1+t]
        ) for i in range(t)]
    rev = np.dot(d, p)
    return d, rev

d, rev = get_demands_rev(a,b,p_data, p)

print('Revenue:')
print(repr_compact.repr(rev))

beta=1.

def get_variance(data_x, p, sigma):
    """
    :param data_x (np.array): [n_samples, n_days]
    :param p (list): [n_days]
    :return: variance
    """
    n_samples, t = data_x.shape
    ones = np.ones((n_samples, 1), dtype= float)
    x_mat = np.concatenate([ones, data_x], axis=1)  # [n_samples, n_days+1]
    x_mat = np.linalg.inv(
        np.dot(x_mat.T, x_mat)
    )
    p = np.array([1.]+p)
    variance = (sigma**2) * (1. + p.dot(x_mat).dot(p))
    return variance

def get_overall_variance(data_x, hist_p, p, sigma):
    all_p = hist_p + p
    t = len(p)
    var = 0
    for i in range(t):
        var += get_variance(data_x, all_p[i+1:i+1+t], sigma)

    return var

objective = rev - beta * get_overall_variance(data_x, p_data, p, sigma)

print('Objective with expected revenue and prediction variance:')
print(repr_compact.repr(objective))

Lp=1e13

for i in range(t):
    penalty_i = ((sum(x[i*n_level:(i+1)*n_level]) - 1)**2)*Lp
    objective -= penalty_i
    
print('Objective with additional equality constraints:')
print(repr_compact.repr(objective))

model = (-objective).compile().to_bqm()

print(repr_compact.repr(model))

print(len(model))

num_shots = 10000

sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample(model, num_reads=num_shots)

# print results
results_df = response.to_pandas_dataframe()
results_columns = results_df.columns[results_df.columns.str.startswith('X_')].tolist()
results_columns += ['energy']

print(results_df[results_columns].sort_values('energy').head())