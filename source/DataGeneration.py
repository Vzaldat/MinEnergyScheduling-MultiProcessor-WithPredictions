import random
import math

def jobsWithoutPred(n, T):
    J = {}
    for i in range(n):
        weight = random.randint(0, 1000)
        arrival = random.randint(0, 3 * T // 4)
        deadline = random.randint(arrival, T)
        J[i] = (weight, arrival, deadline)
    return J

def jobsWithPred(n, T, D, confThres=0.5):
    J = {}
    for i in range(n):
        weight = random.randint(1, 50)
        arrival = random.randint(1, T - D - 1)
        deadline = arrival + D
        
        # Generate prediction confidence from a normal distribution
        # Using confThres as mean and a calculated std_dev to allow for variation
        # Clamp predConf to be between 0 and 1
        std_dev = math.sqrt(confThres * (1 - confThres)) # A common way for bounded values
        if std_dev == 0: # Handle cases where confThres is 0 or 1, which leads to 0 std_dev
            std_dev = 0.01 # Provide a small non-zero std_dev for variation
        
        predConf = random.normalvariate(confThres, std_dev)
        predConf = max(0.0, min(1.0, predConf)) # Clamp to [0, 1]
        
        J[i] = (weight, arrival, deadline, predConf)
    return J