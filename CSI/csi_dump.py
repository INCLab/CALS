import importlib
import pcap
import dpkt
import config
import pandas as pd
import numpy as np
decoder = importlib.import_module(f'decoders.{config.decoder}') # This is also an import

print('Start Sniifing... @ UDP, Port 5500')
sniffer = pcap.pcap(name='wlan0', promisc=True, immediate=True, timeout_ms=50)
sniffer.setfilter('udp and port 5500')

def mac_addr(address):
    return ':'.join('%02X' % dpkt.compat.compat_ord(b) for b in address)

for ts, pkt in sniffer:
    eth = dpkt.ethernet.Ethernet(pkt)
    ip = eth.data 
    udp = ip.data
    csi = udp.data

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
    
    try:
        csi_df.to_csv('outputs.csv')
    except:
        print('Fail to save data')