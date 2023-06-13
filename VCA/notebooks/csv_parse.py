#!/usr/bin/env python 

from multiprocessing import Pool
from vca_helper import *

import tqdm, sys


def run(a): reduce_pcap_csv(**a)


all_sym = glob("/data/jsaxon/vca/ALL-RUNS/*/*/*/*[1-5].csv") + \
          glob("/data/jsaxon/vca/ALL-RUNS/*/*/*/*/*[1-5].csv")

all_sym = sorted(all_sym)

arg_list = []

for f in all_sym:


    computers = "AB"
    if "DOWNLOAD-A" in f: computers = "A"
    if "DOWNLOAD-B" in f: computers = "B"

    for comp in computers:

        arg_list.append({"f" : f, "comp" : comp})


p = Pool(32)
for _ in tqdm.tqdm(p.imap_unordered(run, arg_list),
                   total=len(arg_list)):
    pass

# p.imap_unordered(run, arg_list)

p.close()
p.join()


