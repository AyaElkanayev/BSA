import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import chirp

df = pd.read_csv(r"C:\Users\ayael\OneDrive\Desktop\BSA_2026\spike_times.csv",
                 header=None, names=["spike_time"])
print(df.head())

spikes = df["spike_time"].to_numpy() # spike times in seconds

# Firing rate per 0.01 seconds bin
dt = 0.01
# range = np.arange(spikes.min(), spikes.max()+dt, dt)
range = np.arange(0, 10+dt, dt) # Define the range for the histogram
counts, _ = np.histogram(spikes, bins=range)
spike_rate = counts / dt # Convert counts to firing rate (spikes per second)
t = range[:-1] # Time points for the firing rate


# # 1b
#
# # Original signal
# t = np.linspace(0, 1, 1000)
# signal = chirp(t, f0=10, f1=100, t1=1, method='linear')
#
# # Digital signal - 200 Hz
# fs = 200
# t_digital = np.arange(0, 1, 1 / fs) # 1/200s apart
# signal_digital = chirp(t_digital, f0=10, f1=100, t1=1, method='linear')
#
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title("Original signal (chirp 10-100 Hz)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (a.u.)")
# plt.subplot(2, 1, 2)
# plt.plot(t_digital, signal_digital, marker='o', markersize=3)   # connecting the samples
# plt.title("Digital signal (200 Hz)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude (a.u.)")
# plt.tight_layout()
# plt.show()


# # 1c
#
# # Original signal
# t = np.linspace(0, 1, 1000)
# signal = chirp(t, f0=10, f1=100, t1=1, method='linear')
#
# # 1000 Hz
# t_1000 = np.arange(0, 1, 1/1000)
# s_1000 = chirp(t_1000, f0=10, f1=100, t1=1, method='linear')
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title('Original signal')
# plt.ylabel('Amplitude (a.u.)')
# plt.subplot(2, 1, 2)
# plt.plot(t_1000, s_1000)
# plt.title('1000 Hz')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (a.u.)')
# plt.tight_layout()
# plt.savefig('Q1c.png')
#
# # 150 Hz
# t_150 = np.arange(0, 1, 1/150)
# s_150 = chirp(t_150, f0=10, f1=100, t1=1, method='linear')
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(t, signal)
# plt.title('Original signal')
# plt.ylabel('Amplitude (a.u.)')
# plt.subplot(2, 1, 2)
# plt.plot(t_150, s_150)
# plt.title('150 Hz')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude (a.u.)')
# plt.tight_layout()
# plt.savefig('Q1d.png')
# plt.show()


# 1d


# 2a

plt.figure(figsize=(10, 6))

# Plot 1: spike train
plt.subplot(2, 1, 1)
plt.vlines(spikes, 0, 1)
plt.title("Spike train")
plt.xlabel("Time (s)")
plt.ylabel("Spikes (a.u.)")

# Plot 2: firing rate
plt.subplot(2, 1, 2)
plt.plot(t, spike_rate)
plt.title("Firing rate r(t), 0.01 s bins")
plt.xlabel("Time (s)")
plt.ylabel("Firing rate (Hz)")

plt.tight_layout()
plt.show()

# 2c

dt = 0.01 # bin size in seconds

def gaussian_kernel(sigma, dt):
    # Create a Gaussian kernel with standard deviation sigma
    x = np.arange(-3*sigma, 3*sigma + dt, dt)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum() # Normalize the kernel
    return kernel

short_kernel = gaussian_kernel(0.02, dt)   # sigma = 20 ms
long_kernel = gaussian_kernel(0.1, dt)     # sigma = 100 ms

print(short_kernel.sum(), long_kernel.sum())   # should both print 1.0

smooth_short = np.convolve(spike_rate, short_kernel, mode="same")
smooth_long = np.convolve(spike_rate, long_kernel, mode="same")

plt.subplot(2, 1, 1)
plt.plot(t, smooth_short)
plt.title("Firing rate smoothed with short Gaussian (sigma = 20 ms)")
plt.xlabel("Time (s)")
plt.ylabel("Firing rate (Hz)")

plt.subplot(2, 1, 2)
plt.plot(t, smooth_long)
plt.title("Firing rate smoothed with long Gaussian (sigma = 100 ms)")
plt.xlabel("Time (s)")
plt.ylabel("Firing rate (Hz)")

plt.tight_layout()
plt.show()


# 2d

plt.figure(figsize=(10, 4))
plt.plot(t, spike_rate, label="Original r(t)", alpha=0.5)
plt.plot(t, smooth_long, label="Smoothed (sigma = 100 ms)", linewidth=2)
plt.title("Original vs. smoothed firing rate")
plt.xlabel("Time (s)")
plt.ylabel("Firing rate (Hz)")
plt.legend()
plt.show()


# 2e

sigma = 0.1   # the longer kernel

x = np.arange(0, 3 * sigma + dt, dt)       # only right side
causal_kernel = np.exp(-0.5 * (x / sigma) ** 2)
causal_kernel = causal_kernel / causal_kernel.sum()   # sum = 1 again

smooth_causal = np.convolve(spike_rate, causal_kernel, mode="full")[:len(spike_rate)]

plt.figure(figsize=(10, 4))
plt.plot(t, smooth_long, label="Full Gaussian (symmetric)")
plt.plot(t, smooth_causal, label="Right half only (causal)")
plt.title("Symmetric vs. causal Gaussian kernel (sigma = 100 ms)")
plt.xlabel("Time (s)")
plt.ylabel("Firing rate (Hz)")
plt.legend()
plt.show()

print(causal_kernel.sum())                          # should be 1
print(spike_rate.mean(), smooth_causal.mean())      # about equal
