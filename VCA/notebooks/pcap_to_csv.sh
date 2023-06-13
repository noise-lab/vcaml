#!/bin/bash 

# for f in /data/jsaxon/vca/a/*/captures/*pcap \

for f in $(find /data/vca/Data/competition/ALL-RUNS -name *pcap)
  do

  c=$(echo $f | sed "s/pcap/csv/")
  echo $c
  tshark -r $f -d udp.port==1024-49152,rtp -t e -T fields -e frame.time -e ip.src -e ip.dst -e frame.len > $c

done


