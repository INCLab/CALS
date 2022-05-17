import os
import pandas as pd

# Merge_pe_data 를 이미 거쳤다면 Standard Scaling이 이미 진행된 상태


class DataLoader:

    def loadPEdata(self, dataPath):
        csi_flist = os.listdir(dataPath)

        pe_flist = []
        npe_flist = []
        for file in csi_flist:
            if file.split('_')[0] == 'pe':
                pe_flist.append(os.path.join(dataPath, file))
            elif file.split('_')[0] == 'npe':
                npe_flist.append(os.path.join(dataPath, file))

        # Read csi files and merge with same class
        pe_df = None
        for idx, pe_file in enumerate(pe_flist):
            temp_df = pd.read_csv(pe_file)
            if idx == 0:
                pe_df = temp_df
            else:
                pe_df = pd.concat([pe_df, temp_df], ignore_index=True)

        npe_df = None
        for idx, npe_file in enumerate(npe_flist):
            temp_df = pd.read_csv(npe_file)
            if idx == 0:
                npe_df = temp_df
            else:
                npe_df = pd.concat([npe_df, temp_df], ignore_index=True)

        return pe_df, npe_df