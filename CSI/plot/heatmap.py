import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot.dataPreprocess import data_preprocess

'''
Heatmap
---------------------------

Plot 
'''


def heatmap(csi_df, sample_start, sample_end, preprocess=False):

    df = csi_df[sample_start:sample_end]

    if preprocess is True:
        df = data_preprocess(df)

    plt.pcolor(df)

    plt.title('CSI heatmap')
    plt.xlabel('Subcarrier')
    plt.ylabel('Packets')
    plt.colorbar()
    plt.show()


