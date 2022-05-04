'''
    Extract raw CSI data (complex number)
'''

import pcap
import dpkt
import keyboard
import pandas as pd
import numpy as np
import os
from datetime import datetime
import time


BANDWIDTH = 20

# number of subcarrier
NSUB = int(BANDWIDTH * 3.2)


# for sampling
def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)


def sniffing(nicname):
    print('Start Sniifing... @', nicname, 'UDP, Port 5500')
    sniffer = pcap.pcap(name=nicname, promisc=True, immediate=True, timeout_ms=50)
    sniffer.setfilter('udp and port 5500')
    
    column = ['mac', 'time'] + ['_' + str(i) for i in range(0, NSUB)]

    # Dataframe by mac address
    mac_dict = {}

    before_ts = 0.0

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

        # 해당 mac address 키 값이 없을 경우 새로운 dataframe 생성 후 dict에 추가
        if mac not in mac_dict:
            mac_dict[mac] = pd.DataFrame(columns=column)

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

        csi_df = pd.DataFrame(csi_cmplx)
        csi_df.insert(0, 'mac', mac)
        csi_df.insert(1, 'time', ts)

        # Rename Subcarriers Column Name
        columns = {}
        for i in range(0, 64):
            columns[i] = '_' + str(i)

        csi_df.rename(columns=columns, inplace=True)

        # Save dataframe to SQL
        try:
            mac_dict[mac] = pd.concat([mac_dict[mac], csi_df], ignore_index=True)
        except Exception as e:
            print('Error', e)

        before_ts = ts

        if keyboard.is_pressed('s'):
            print("Stop Collecting...")

            for mac_address in mac_dict.keys():
                mac_dict[mac_address].to_csv('csi_{}_{}MHz.csv'.format(mac_address, bandwidth), index=False)
            break


def ping(nicname):
    print('Start Ping...')

    # Get Gateway IP
    gwipcmd = "ip route | grep -w 'default via.*dev " + nicname + "' | awk '{print $3}'"
    gwip = os.popen(gwipcmd).read()

    # Send Ping
    while True:
        # Request 5 Times, Ping from specified NIC to gateway
        pingcmd = 'ping -q -c 5 -I ' + nicname + ' ' + gwip + ' 1> /dev/null'
        os.system(pingcmd)

        # Sleep
        time.sleep(1)


if __name__ == '__main__':
    sniffing('wlan0')
    # CSI Extractor Interface
    # csinicname = 'wlan1'
    #
    # # Ping dedicated interface
    # pingnicname = 'wlan0'
    #
    # sniffing = Process(target=sniffing, args=(csinicname, ))
    # ping = Process(target=ping, args=(pingnicname, ))
    #
    # sniffing.start()
    # ping.start()
    #
    # sniffing.join()
    # ping.join()