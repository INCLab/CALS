import pcap
import dpkt
import pandas as pd
import numpy as np
from database.tracking_db import tracking_db
from multiprocessing import Process
import os
import time
def sniffing(nicname):
    db = tracking_db()

    print('Start Sniifing... @', nicname, 'UDP, Port 5500')
    sniffer = pcap.pcap(name=nicname, promisc=True, immediate=True, timeout_ms=50)
    sniffer.setfilter('udp and port 5500')

    for ts, pkt in sniffer:
        eth = dpkt.ethernet.Ethernet(pkt)
        ip = eth.data
        udp = ip.data
        csi = udp.data[18:]

        bandwidth = ip.__hdr__[2][2]
        nsub = int(bandwidth * 3.2)

        # Convert CSI bytes to numpy array
        csi_np = np.frombuffer(
            csi,
            dtype = np.int16,
            count = nsub * 2
        )

        # Cast numpy 1-d array to matrix
        csi_np = csi_np.reshape((1, nsub * 2))

        # Convert csi into complex numbers
        csi_cmplx = np.fft.fftshift(
            csi_np[:1, ::2] + 1.j * csi_np[:1, 1::2], axes=(1,)
        )

        csi_df = pd.DataFrame(np.abs(csi_cmplx))
        csi_df.insert(0, 'time', ts)

        # Rename Subcarriers Column Name
        columns = {}
        for i in range(0, 64):
            columns[i] = '_' + str(i)

        csi_df.rename(columns=columns, inplace=True)

        # Save dataframe to SQL
        try:
            db.insert_csi(csi_df)
        except Exception as e:
            print('Error', e)

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
    # CSI Extractor Interface
    csinicname = 'wlan1'

    # Ping dedicated interface
    pingnicname = 'wlan0'

    sniffing = Process(target=sniffing, args=(csinicname, ))
    ping = Process(target=ping, args=(pingnicname, ))

    sniffing.start()
    ping.start()

    sniffing.join()
    ping.join()