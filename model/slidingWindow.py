import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from hampel import hampel


def makeSlidingWindow(df, window_size, subcarrier, filter):
    s_df = df[[subcarrier, 'label']]

    if filter is True:
        signal = s_df[subcarrier].to_list()

        # Outlier Imputation with rolling median
        ts_imputation = hampel(pd.Series(signal), window_size=5, n=3, imputation=True)
        s_df[subcarrier] = ts_imputation
        #
        # print(len(ts_imputation))
        # print(len(s_df[subcarrier]))
        # s_df[subcarrier].plot(style="k-")
        # ts_imputation.plot(style="g-")
        # plt.show()

        # fig, ax = plt.subplots(figsize=(12, 8))
        # label = s_df['label'][0]
        # fig.suptitle('Amp-SampleIndex plot Label:{}'.format(label))
        # ax.plot(signal, color="b", alpha=0.5)
        # rec = lowpassfilter(signal, 0.4)
        # ax.plot(rec, 'k', label='DWT smoothing}', linewidth=2)
        # ax.legend()
        # ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
        # ax.set_ylabel('Signal Amplitude', fontsize=16)
        # ax.set_xlabel('Sample Index', fontsize=16)
        # plt.show()

        # filtered_signal = lowpassfilter(signal, 0.4)
        # print(len(s_df[subcarrier]))
        # print(len(filtered_signal))
        # s_df[subcarrier] = filtered_signal

    label_list = s_df['label'].drop_duplicates().to_list()

    sliding_list = []
    for label in label_list:
        tmp_df = s_df[s_df['label'] == label]

        subc_list = tmp_df[subcarrier].to_list()

        total_second = int(len(subc_list) / window_size)
        start_num = 0
        end_num = window_size

        while True:
            end_num = start_num + window_size
            if end_num > total_second * window_size:
                break
            current_csi = subc_list[start_num:end_num]
            current_csi.append(label)
            sliding_list.append(current_csi)
            start_num += int(window_size/2)

    column_list = [str(i) for i in range(1, window_size + 1)] + ['label']
    sliding_df = pd.DataFrame(sliding_list, columns=column_list)

    return sliding_df


def lowpassfilter(signal, thresh=0.63, wavelet="db4"):
    thresh = thresh * np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per", level=8)
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal