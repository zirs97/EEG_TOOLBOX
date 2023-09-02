from mne.time_frequency import psd_array_welch
import math
import numpy as np


def compute_psd_de(data, freq_bands, sampling_rate):  # TODO: 可能存在问题 不要近似计算de
    n_channels, datapoints = data.shape
    psd_all, de_all = np.zeros((n_channels, len(freq_bands))), np.zeros(
        (n_channels, len(freq_bands)))
    for channel in range(n_channels):
        segment = data[channel, :]
        psd_bands, freqs = psd_array_welch(
            segment, sampling_rate, fmin=freq_bands[0][0], fmax=freq_bands[-1][1])
        for band, (f_min, f_max) in enumerate(freq_bands):
            freq_mask = (freqs >= f_min) & (freqs <= f_max)
            psd_band = psd_bands[freq_mask]
            psd = np.sum(psd_band)

            # de = differential_entropy(segment)
            # de = math.log(psd, 2) # 近似简化 但是还是不近似比较好
            de = math.log2(psd + 1)  # 保持非负

            psd_all[channel, band] = psd
            de_all[channel, band] = de

    return psd_all, de_all
