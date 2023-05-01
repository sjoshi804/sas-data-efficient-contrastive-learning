import numpy as np 
import random

def save_np_array(template_file_name: str, array_label: str, np_array: np.array):
    with open(f"{template_file_name}-{array_label}.npy", 'wb+') as csvfile:
        np.save(csvfile, np_array, allow_pickle=False, fix_imports=False)
    
class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Random(metaclass=Singleton):
    def __init__(self, seed=0):
        self.random = random 
        self.random.seed(seed)