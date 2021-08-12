import os,sys
import shutil
import time
import traceback
import numpy as np
import sklearn.metrics as skm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

lengths = []
with open(r"../../ev_lengths.txt","r") as infile:
    line = infile.readline()
    count = 1
    while line != "\n":
        # print("count: ",count)
        if count >= 9:
            # print("do stuff")
            print("line:",line)
            line = line[:-1]
            lengths.append(int(line))            
        line = infile.readline()
        count += 1
plt.clf()
plt.hist(lengths, bins=100, histtype="step")
plt.xlabel("Entry Lengths (pixels)")
plt.ylabel("Number per bin")
plt.title("Length of Entry")
# plt.legend()
plt.tight_layout()
plt.savefig("histograms/entry_lengths.png")
plt.close()
