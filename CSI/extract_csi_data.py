import importlib

from scapy.all import *
import config
import pandas as pd
import numpy as np
from database.tracking_db import tracking_db

decoder = importlib.import_module(f'decoders.{config.decoder}') # This is also an import

db = tracking_db()

def func(pkt):
    global limit, count, timestamps
    timestamps.append(pkt.time)
    count = count + 1

    if count >= limit > 0:
        return True
    else:
        return False


if __name__ == "__main__":
    limit = 26000000
    count = 0
    timestamps = []
    filename = "../data/class_0.pcap"

    # Read pcap file and create dataframe
    try:
        csi_samples = decoder.read_pcap(filename)
    except FileNotFoundError:
        print(f'File {filename} not found.')
        exit(-1)

    # Create CSI data frame
    csi_df = pd.DataFrame(np.abs(csi_samples.get_all_csi()))

    sniff(offline=filename, stop_filter=func, store=False)

    # Decimal to Float
    floatts = list(map(float, timestamps))

    # Insert Timestamps
    csi_df.insert(0, 'time', floatts)

    # Rename Subcarriers Column Name
    columns = {}
    for i in range(0, 64):
        columns[i] = '_' + str(i)

    csi_df.rename(columns=columns, inplace=True)

    # Save dataframe to SQL
    try:
        db.insert_csi(csi_df)
        print('Success to Save Data')
    except Exception as e:
        print('Fail to save data\n', e)
