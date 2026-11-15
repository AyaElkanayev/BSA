import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import chirp

def generatePoiSpikes(r, dt, totalSize, tRef=0):
    # Poisson spike train. r [spikes/s], dt [s], totalSize [bins], tRef [s]
    # Returns spike times in ms
    # (a) homogeneous Poisson: each bin spikes with probability r*dt (vectorized)
    spikeTrain = np.flatnonzero(np.random.rand(totalSize) < r * dt) * (dt * 1000)
    if tRef == 0:
        return spikeTrain
    # (b) refractory period: keep each candidate spike with probability r(t)/r0,
    #  where r(t) = 0 during tRef, then r0*(t - tLast)/5, capped at r0
    kept = []
    tLast = -np.inf
    for t in spikeTrain:
        sinceLast = t - tLast   # ms
        if sinceLast <= tRef * 1000:
            continue
        if np.random.rand() < min(1, sinceLast / 5):
            kept.append(t)
            tLast = t
    return np.array(kept)

def calcFF(spikeTrain, winLen, duration):
    # Fano factor of a single train: cut it into windows of winLen [s]
    # and compute var/mean of the spike counts. duration in s
    edges = np.arange(0, duration + 1e-9, winLen) * 1000
    counts, _ = np.histogram(spikeTrain, edges)
    return counts.var() / counts.mean()

def calcCV(spikeTrain):
    # CV of the inter-spike intervals
    isi = np.diff(spikeTrain)
    return isi.std() / isi.mean()