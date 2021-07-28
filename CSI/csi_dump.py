import pcap
import dpkt

sniffer = pcap.pcap(name='wlan0', promisc=True, immediate=True, timeout_ms=50)
sniffer.setfilter('udp and port 5500')

def mac_addr(address):
    return ':'.join('%02X' % dpkt.compat.compat_ord(b) for b in address)

for ts, pkt in sniffer:
    eth = dpkt.ethernet.Ethernet(pkt)
    print(ts, mac_addr(eth.dst), mac_addr(eth.src), eth.type)
    print(eth.data)
