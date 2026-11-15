import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.io import loadmat
from spike_tools import generatePoiSpikes, calcFF, calcCV

# t = np.linspace(0, 1, 1000) # Define time array

# # Generate the signal, including frequencies of 10 to100 Hz
# signal = chirp(t, f0=10, f1=100, t1=1, method='linear')
# print(signal)

np.random.seed(3)

# 2a
r0, dt, T = 30, 0.001, 60
nBins = int(T / dt)
trains = {
    "Poisson": generatePoiSpikes(r0, dt, nBins),
    "Poisson + refractory": generatePoiSpikes(r0, dt, nBins, tRef=0.001),
    "rawSpikes1": loadmat("rawSpikes1.mat")["spikes"].ravel(),
    "rawSpikes2": loadmat("rawSpikes2.mat")["spikes"].ravel(),
}
# recording length in s (time of the last spike)
durations = {name: T if name.startswith("Poisson") else st[-1] / 1000
             for name, st in trains.items()}

# i
fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
for ax, (name, st) in zip(axs, trains.items()):
    ax.eventplot(st[st < 1000])
    ax.set_title(name, fontsize=9, loc="left")
    ax.set_yticks([])
axs[-1].set_xlabel("Time [ms]")
fig.tight_layout()
fig.savefig("fig1_spike_trains.png", dpi=150)

# ii
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for ax, (name, st) in zip(axs.ravel(), trains.items()):
    ax.hist(np.diff(st), bins=np.arange(0, 151), density=True)
    ax.set_title(name)
    ax.set_xlabel("ISI [ms]")
    ax.set_ylabel("Probability density")
fig.tight_layout()
fig.savefig("fig2_TIH.png", dpi=150)

# iii and iv
print(f"{'train':22s} FF(0.1s) FF(1s) FF(3s)    CV  CV^2")
for name, st in trains.items():
    ff = [calcFF(st, w, durations[name]) for w in (0.1, 1, 3)]
    cv = calcCV(st)
    print(f"{name:22s} {ff[0]:7.3f} {ff[1]:6.3f} {ff[2]:6.3f} {cv:5.3f} {cv**2:5.3f}")