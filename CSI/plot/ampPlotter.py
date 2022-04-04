import numpy as np
import matplotlib.pyplot as plt
import pywt
from plot.dataPreprocess import data_preprocess

'''
Time plotter
---------------------------

Plot 
'''


def lowpassfilter(signal, thresh=0.63, wavelet="db4"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per", level=8)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal


def AmpPlotter(csi_df, sample_start, sample_end):

    csi_df = csi_df[sample_start:sample_end]

    subcarrier = csi_df['sub7'].to_list()
    subcarrier2 = csi_df['sub10'].to_list()
    subcarrier3 = csi_df['sub1'].to_list()

    # ============ Denoising with DWT ==================
    signal = subcarrier
    signal2 = subcarrier2
    signal3 = subcarrier3

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Time-Subcarrier plot')
    ax.plot(signal, color="b", alpha=0.5, label='sub7')
    ax.plot(signal2, color="r", alpha=0.5, label='sub10')
    ax.plot(signal3, color="y", alpha=0.5, label='sub13')
    rec = lowpassfilter(signal, 0.2)
    ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
    ax.legend()
    ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
    ax.set_ylabel('Signal Amplitude', fontsize=16)
    ax.set_xlabel('Sample No', fontsize=16)
    plt.show()


