import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
t = np.linspace(0, 1, 1000) # Define time array


# Generate the signal, including frequencies of 10 to100 Hz
signal = chirp(t, f0=10, f1=100, t1=1, method='linear')
print(signal)

