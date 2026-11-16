import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Q1

# 1a
def generatePoiSpikes(r, dt, totalSize, tRef=0):
    spikeTrain = np.flatnonzero(np.random.rand(totalSize) < r * dt) * (dt * 1000) # in ms
    if tRef == 0: # vectorized
        return spikeTrain

# 1b rate must be 0 at tRef, then climb back to r0*(t-t_spike)/5
    kept = []
    tLast = -np.inf
    for t in spikeTrain:
        Last = t - tLast       # ms
        if Last <= tRef * 1000:
            continue
        if np.random.rand() < min(1, Last/5):
            kept.append(t)
            tLast = t
    return np.array(kept)

# 1c FF - single spike train
def calcFF(spikeTrain, windows, duration): # recording cut into windows
    # compute var/mean of the spike counts
    edges = np.arange(0, duration + 1e-9, windows) * 1000 # adding a tiny number moves the stop just past 60
    counts, _ = np.histogram(spikeTrain, edges)
    return counts.var()/counts.mean()

# 1d CV of ISIs
def calcCV(spikeTrain):
    isi = np.diff(spikeTrain)
    return isi.std()/isi.mean()


# Q2
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

# recording length in s
durations = {name: T if name.startswith("Poisson") else st[-1] / 1000
             for name, st in trains.items()}

# i first second
fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
for ax, (name, st) in zip(axs, trains.items()):
    ax.eventplot(st[st < 1000])
    ax.set_title(name, fontsize=9, loc="left")
    ax.set_yticks([])
axs[-1].set_xlabel("Time [ms]")
fig.tight_layout()
fig.savefig("2a_i_spike_trains.png", dpi=150)

# ii
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for ax, (name, st) in zip(axs.ravel(), trains.items()):
    ax.hist(np.diff(st), bins=np.arange(0, 151), density=True)
    ax.set_title(name)
    ax.set_xlabel("ISI [ms]")
    ax.set_ylabel("Probability density")
fig.tight_layout()
fig.savefig("2a_ii_TIH.png", dpi=150)

# iii and iv
print(f"{'train':22s} FF(0.1s) FF(1s) FF(3s)    CV  CV^2")
for name, st in trains.items():
    ff = [calcFF(st, w, durations[name]) for w in (0.1, 1, 3)]
    cv = calcCV(st)
    print(f"{name:22s} {ff[0]:7.3f} {ff[1]:6.3f} {ff[2]:6.3f} {cv:5.3f} {cv**2:5.3f}")