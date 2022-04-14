import pcap
import dpkt
import keyboard
import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Process

# : 제외
selected_mac = 'dca6328e1dcb'
show_packet_length = 100

global xval
xval = 0

# for sampling
def truncate(num, n):
    integer = int(num * (10 ** n)) / (10 ** n)
    return float(integer)


def sniffing(nicname, mac_address):
    print('Start Sniifing... @', nicname, 'UDP, Port 5500')
    sniffer = pcap.pcap(name=nicname, promisc=True, immediate=True, timeout_ms=50)
    sniffer.setfilter('udp and port 5500')

    before_ts = 0.0

    column = ['mac', 'time'] + ['_' + str(i) for i in range(0, 64)]

    result_df = pd.DataFrame(columns=column)

    for ts, pkt in sniffer:
        if int(ts) == int(before_ts):
            cur_ts = truncate(ts, 1)
            bef_ts = truncate(before_ts, 1)

            if cur_ts == bef_ts:
                before_ts = ts
                continue

        eth = dpkt.ethernet.Ethernet(pkt)
        ip = eth.data
        udp = ip.data

        # MAC Address 추출
        # UDP Payload에서 Four Magic Byte (0x11111111) 이후 6 Byte는 추출된 Mac Address 의미
        mac = udp.data[4:10].hex()

        if mac != mac_address:
            continue


        # Four Magic Byte + 6 Byte Mac Address + 2 Byte Sequence Number + 2 Byte Core and Spatial Stream Number + 2 Byte Chanspac + 2 Byte Chip Version 이후 CSI
        # 4 + 6 + 2 + 2 + 2 + 2 = 18 Byte 이후 CSI 데이터
        csi = udp.data[18:]

        bandwidth = ip.__hdr__[2][2]
        nsub = int(bandwidth * 3.2)

        # Convert CSI bytes to numpy array
        csi_np = np.frombuffer(
            csi,
            dtype=np.int16,
            count=nsub * 2
        )

        # Cast numpy 1-d array to matrix
        csi_np = csi_np.reshape((1, nsub * 2))

        # Convert csi into complex numbers
        csi_cmplx = np.fft.fftshift(
            csi_np[:1, ::2] + 1.j * csi_np[:1, 1::2], axes=(1,)
        )

        csi_df = pd.DataFrame(np.abs(csi_cmplx))
        csi_df.insert(0, 'mac', mac)
        csi_df.insert(1, 'time', ts)

        # Rename Subcarriers Column Name
        columns = {}
        for i in range(0, 64):
            columns[i] = '_' + str(i)

        csi_df.rename(columns=columns, inplace=True)

        try:
            result_df = pd.concat([result_df, csi_df], ignore_index=True)
            print(result_df)

            if len(result_df) >= show_packet_length:
                result_df = result_df[1:]

        except Exception as e:
            print('Error', e)

        before_ts = ts

        if keyboard.is_pressed('s'):
            print("Stop Collecting...")



    subcarrier_list = []
    for col in result_df.columns:
        subcarrier_list.append(result_df[col].to_list())
    x = []
    y_list = [[] for i in range(0, len(subcarrier_list))]

    def animate(frame):

        # 100packet 이상 넘어가면 맨 앞 packet 제거
        if len(x) > show_packet_length:
            del x[0]
            for y in y_list:
                del y[0]

        global xval
        x.append(xval)
        for j in range(0, len(subcarrier_list)):
            y_list[j].append(subcarrier_list[j][xval])
        plt.cla()
        for idx, y in enumerate(y_list):
            plt.plot(x, y, alpha=0.5, label=csi_df.columns[idx])

        plt.title(mac_address)
        plt.ylabel('Signal Amplitude', fontsize=16)
        plt.xlabel('Packet', fontsize=16)
        plt.tight_layout()
        xval += 1

    ani = FuncAnimation(plt.gcf(), animate, interval=10, frames=240, repeat=False)
    # ani.save(r'csi.gif', fps=10)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sniffing('wlan0', selected_mac)
    # CSI Extractor Interface
    # csinicname = 'wlan1'
    #
    # # Ping dedicated interface
    # pingnicname = 'wlan0'
    #
    # sniffing = Process(target=sniffing, args=('wlan0', selected_mac, ))
    # rtPlot = Process(target=AmpRTPlotter, args=(selected_mac, ))
    #
    # sniffing.start()
    # rtPlot.start()
    #
    # sniffing.join()
    # ping.join()