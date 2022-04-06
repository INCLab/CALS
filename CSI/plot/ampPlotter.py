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


def AmpPlotter(csi_df, sample_start, sample_end, spf_sub=None):

    csi_df = csi_df[sample_start:sample_end]

    if spf_sub is not None:
        subcarrier = csi_df[spf_sub].to_list()

        # ============ Denoising with DWT ==================
        signal = subcarrier

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Time-Subcarrier plot')
        ax.plot(signal, color="b", alpha=0.5, label=spf_sub)
        rec = lowpassfilter(signal, 0.2)
        ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
        ax.legend()
        ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
        ax.set_ylabel('Signal Amplitude', fontsize=16)
        ax.set_xlabel('Sample No', fontsize=16)
        plt.show()
    else:
        subcarrier_list = []
        for col in csi_df.columns:
            subcarrier_list.append(csi_df[col].to_list())

        # ============ Denoising with DWT ==================

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Time-Subcarrier plot')

        for idx, sub in enumerate(subcarrier_list):
            ax.plot(sub, alpha=0.5, label=csi_df.columns[idx])

        ax.legend()
        ax.set_ylabel('Signal Amplitude', fontsize=16)
        ax.set_xlabel('Sample No', fontsize=16)
        plt.show()


def AmpTimePlotter(csi_df, sample_start, sample_end, spf_sub=None):

    csi_df = csi_df[sample_start:sample_end]

    if spf_sub is not None:
        subcarrier = csi_df[spf_sub].to_list()

        # ============ Denoising with DWT ==================
        signal = subcarrier

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Time-Subcarrier plot')
        ax.plot(signal, color="b", alpha=0.5, label=spf_sub)
        rec = lowpassfilter(signal, 0.2)
        ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
        ax.legend()
        ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
        ax.set_ylabel('Signal Amplitude', fontsize=16)
        ax.set_xlabel('Sample No', fontsize=16)
        plt.show()
    else:
        subcarrier_list = []
        for col in csi_df.columns:
            subcarrier_list.append(csi_df[col].to_list())

        # ============ Denoising with DWT ==================

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Time-Subcarrier plot')

        for idx, sub in enumerate(subcarrier_list):
            ax.plot(sub, alpha=0.5, label=csi_df.columns[idx])

        ax.legend()
        ax.set_ylabel('Signal Amplitude', fontsize=16)
        ax.set_xlabel('Sample No', fontsize=16)
        plt.show()

