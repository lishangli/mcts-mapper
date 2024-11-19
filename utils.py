import numpy as np

class Action:
    def __init__(self, id, time):
        self.id = id
        self.time = time

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()