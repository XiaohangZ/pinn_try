from math import sqrt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def accuracy(predictions,labels):
    # print("mean_absolute_error:", mean_absolute_error(predictions, labels))
    # print("mean_squared_error:", mean_squared_error(predictions, labels))
    # print("rmse:", sqrt(mean_squared_error(predictions, labels)))
    # print("r2 score:", r2_score(predictions, labels))
    return mean_squared_error(predictions, labels)


# b = np.array( [ (1.5,2), (4,5), (6,7) ]  )
# c = np.array( [ (2,8), (4.5,5.5), (6.5,7.5) ]  )
# accuracy(b,c)

