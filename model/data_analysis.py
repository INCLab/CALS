import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def dataAnalysisPE(df):

    print("\n < Data info >")
    print(df.info())

    print("\n < Data Describe >")
    print(df.describe())

    PE_len = len(df[df['label'] == 1])
    NPE_len = len(df[df['label'] == 0])

    colors = sns.color_palette('hls', 2)

    plt.bar([1, 2], [PE_len, NPE_len], width=0.6, color=colors)
    plt.xlabel('Label', fontsize=18)
    plt.ylabel('Amount of Data', fontsize=18)
    plt.xticks([1, 2], ['Presence', 'None'])
    plt.show()
