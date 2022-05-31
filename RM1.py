import sys

import numpy as np
import pandas as pd

sys.path.append('../')
from revpy.revpy import protection_levels
from revpy.helpers import cumulative_booking_limits
from revpy.fare_transformation import calc_fare_transformation

fares = np.array([1200, 1000, 800, 600, 400, 200])
demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])
sigmas = np.array([11.2, 6.6, 7.7, 8.9, 10.4, 12])
capacity = 100
classes = np.arange(len(fares)) + 1
class_names = ["class_{}".format(c) for c in classes]

adjusted_fares, adjusted_demand, Q, TR = calc_fare_transformation(fares, demands, return_all=True)

data = np.transpose(np.vstack((fares, demands, Q, TR, adjusted_fares)))
pd.DataFrame(data, index=class_names, columns=['fares', 'demand', 'Q', 
                                               'TR', 'MR/adjusted fares'])