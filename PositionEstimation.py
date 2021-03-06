import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

from lstm import LSTM

data = pd.read_excel("data.xlsx")
data = data.values



# data = scale(data, axis=0)

a = np.diff(data[:, range(3, 5)], axis=0)
step = 25
input = []
target = []
for i in range(step, a.shape[0]):
    input.append(a[range(i - step, i - 1), :])
    target.append(data[i, [1, 2]])



lstm = LSTM()
lstm.Train(input, target)
