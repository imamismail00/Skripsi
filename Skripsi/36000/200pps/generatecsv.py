import sys
import getopt
import logging
import numpy as np
import pickle
import os
import pandas as pd
import csv

def generatecsv(input_file, data):
    c = input_file  # Corrected variable name from `input` to `input_file`
    with open(c, "a") as f:
        fnames = [
            "datapath_id",
            "version",
            "header_length",
            "tos",
            "total_length",
            "flags",
            "offset",
            "ttl",
            "proto",
            "csum",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "tcp_flag",
            "type_icmp",
            "code_icmp",
            "csum_icmp",
            "port_no",
            "label",
        ]
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        for a in data:
            temp = a.rstrip("\n").split(";")
            writer.writerow({
                "datapath_id": temp[0],
                "version": temp[1],
                "header_length": temp[2],
                "tos": temp[3],
                "total_length": temp[4],
                "flags": temp[5],
                "offset": temp[6],
                "ttl": temp[7],
                "proto": temp[8],
                "csum": temp[9],
                "src_ip": temp[10],
                "dst_ip": temp[11],
                "src_port": temp[12],
                "dst_port": temp[13],
                "tcp_flag": temp[14],
                "type_icmp": temp[15],
                "code_icmp": temp[16],
                "csum_icmp": temp[17],
                "port_no": temp[18],
                "label": temp[19],
            })

data2 = []
with open('ensemble_boosting', 'r') as file:
    data2 = file.readlines()

generatecsv("ensemble_boosting.csv", data2)

