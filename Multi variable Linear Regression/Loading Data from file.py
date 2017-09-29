import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

xy = np.loadtxt('data-01-test-score.csv', delimiter=',',dtype=np.float32)
x_data = xy[:, 0:-1] # Slicing
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)