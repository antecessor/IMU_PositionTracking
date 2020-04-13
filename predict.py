import pandas as pd

import numpy as np
from tensorflow_core.python.keras.saving import load_model

model=load_model("PositionEstimation.h5")
data = pd.read_excel("data.xlsx")
data = data.values

a = np.diff(data[:, range(3, 5)], axis=0)
step = 25
input = []
target = []
for i in range(step, a.shape[0]):
    input.append(a[range(i - step, i - 1), :])
    target.append(data[i, [1, 2]])

input = np.reshape(input, [-1, input[0].shape[0], input[0].shape[1]])
predicted=model.predict(input)
target = np.asarray(target)

res = pd.DataFrame({"predicted_x": predicted[:, 0],
                    "predicted_y": predicted[:, 1],
                    "original_x": target[:, 0],
                    "original_y": target[:, 1]})
res.to_excel("res.xlsx")