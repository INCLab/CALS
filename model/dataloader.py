import os
import pandas as pd
from slidingWindow import makeSlidingWindow

# Merge_pe_data 를 이미 거쳤다면 Standard Scaling이 이미 진행된 상태


class DataLoader:

    def loadPEdata(self, dataPath):
        pe_flist, npe_flist = self.__createFileList(dataPath)

        # Read csi files and merge with same class
        pe_df = self.__createDataFrame(pe_flist)
        npe_df = self.__createDataFrame(npe_flist)

        return pe_df, npe_df


    def loadWindowPeData(self, dataPath):
        pe_flist, npe_flist = self.__createFileList(dataPath)

        pe_df = self.__createSlidingDF(pe_flist, 'pe')
        npe_df = self.__createSlidingDF(npe_flist, 'npe')

        return pe_df, npe_df


    def __createFileList(self, dataPath):
        csi_flist = os.listdir(dataPath)

        pe_flist = []
        npe_flist = []
        for file in csi_flist:
            if file.split('_')[0] == 'pe':
                pe_flist.append(os.path.join(dataPath, file))
            elif file.split('_')[0] == 'npe':
                npe_flist.append(os.path.join(dataPath, file))

        return pe_flist, npe_flist

    def __createDataFrame(self, flist):
        df = None
        for idx, file in enumerate(flist):
            temp_df = pd.read_csv(file)
            if idx == 0:
                df = temp_df
            else:
                df = pd.concat([df, temp_df], ignore_index=True)
        return df

    def __createSlidingDF(self, flist, isPE):
        df = None
        for idx, file in enumerate(flist):
            csi_df = pd.read_csv(file)
            subcarrier_list = csi_df.columns.to_list()[2:-1]

            if isPE == 'pe':
                # PE data에서 label이 0인경우 삭제
                indexNames = csi_df[csi_df['label'] == 0].index
                # Delete these row indexes from dataFrame
                csi_df.drop(indexNames, inplace=True)
            else:
                # NPE data에서 label이 1인경우 삭제
                indexNames = csi_df[csi_df['label'] == 1].index
                # Delete these row indexes from dataFrame
                csi_df.drop(indexNames, inplace=True)

            sliding_df = None
            for i, subcarrier in enumerate(subcarrier_list):
                temp_df = makeSlidingWindow(csi_df, subcarrier)
                if i == 0:
                    sliding_df = temp_df
                else:
                    sliding_df = pd.concat([sliding_df, temp_df], ignore_index=True)

            if idx == 0:
                df = sliding_df
            else:
                df = pd.concat([df, sliding_df], ignore_index=True)

        return df

if __name__ == "__main__":
    pe_df, npe_df = DataLoader().loadWindowPeData('../data/pe')