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
