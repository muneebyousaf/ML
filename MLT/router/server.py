


import socket
import sys

from scapy.all import *

def custom_action(Packet):
    
    print(Packet.summary())
    if IP in Packet:
        ip_src=Packet[IP].src
        print(ip_src)
        ip_dst=Packet[IP].dst
        print(ip_dst)
        Packet[IP].dst= '10.42.0.169'
        Packet[IP].src= '10.42.0.169'
        
        sendp(Packet)

        Packet.show()

'''
  # uploadPacket function has access to the url & token parameters
  # because they are 'closed' in the nested function
  def upload_packet(packet):
    # upload packet, using passed arguments
    headers = {'content-type': 'application/json'}
    data = {
        'packet': packet.summary(),
        'token': token,
    }
   # r = requests.post(url, data=data, headers=headers)

  return NULL

sniff(prn=custom_action(url, token))
a=sniff(count=10)
print(a.nsummary())
'''

sniff(prn=custom_action)
#sniff(prn=lambda x:x.summary(), count=5)

exit(0)

if len(sys.argv) == 3:
    # Get "IP address of Server" and also the "port number" from
    ip = sys.argv[1]
    port = int(sys.argv[2])
else:
    print("Run like : python3 server.py <arg1:server ip:this system IP 192.168.1.6> <arg2:server port:4444 >")
    exit(1)

# Create a UDP socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind the socket to the port
server_address = (ip, port)
s.bind(server_address)
print("Do Ctrl+c to exit the program !!")

while True:
    print("####### Server is listening #######")
    data, address = s.recvfrom(4096)
    print("\n\n 2. Server received: ", data.decode('utf-8'), "\n\n")
    send_data = input("Type some text to send => ")
    s.sendto(send_data.encode('utf-8'), address)
    print("\n\n 1. Server sent : ", send_data,"\n\n")


'''
    #!/usr/bin/env python
from scapy.all import *
def print_summary(pkt):
    if IP in pkt:
        ip_src=pkt[IP].src
        ip_dst=pkt[IP].dst
    if TCP in pkt:
        tcp_sport=pkt[TCP].sport
        tcp_dport=pkt[TCP].dport

        print " IP src " + str(ip_src) + " TCP sport " + str(tcp_sport) 
        print " IP dst " + str(ip_dst) + " TCP dport " + str(tcp_dport)

    # you can filter with something like that
    if ( ( pkt[IP].src == "192.168.0.1") or ( pkt[IP].dst == "192.168.0.1") ):
        print("!")

sniff(filter="ip",prn=print_summary)
# or it possible to filter with filter parameter...!
sniff(filter="ip and host 192.168.0.1",prn=print_summary)

'''