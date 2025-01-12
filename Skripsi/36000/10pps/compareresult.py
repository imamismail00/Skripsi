import sys
import getopt
import logging
import numpy as np
import pickle
import os
import pandas as pd

def getindex(input, output, outputs):
    filepath1 = input
    filepath2 = "36000.csv"

    testsdn = pd.read_csv(filepath1)
    test = pd.read_csv(filepath2)

    # Remove rows with null values
    testsdn = testsdn.dropna()
    test = test.dropna()

    testsdn = testsdn.drop_duplicates()
    test = test.drop_duplicates()

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    x_sdn = testsdn.drop(columns=['label'])
    y_sdn = testsdn['label'].values

    x_test = test.drop(columns=['rx_bytes_ave','rx_error_ave','rx_dropped_ave','tx_bytes_ave', 'tx_error_ave', 'tx_dropped_ave', 'label'])
    y_test = test['label'].values

    y_test = le.fit_transform(y_test)

    data = np.array([])
    datatemp = np.array([])
    i = 2

    for a in x_sdn.index:
        index_list = x_test[(x_test['datapath_id'] == x_sdn.at[a, 'datapath_id']) &
                            (x_test['version'] == x_sdn.at[a, 'version']) &
                            (x_test['header_length'] == x_sdn.at[a, 'header_length']) &
                            (x_test['tos'] == x_sdn.at[a, 'tos']) &
                            (x_test['total_length'] == x_sdn.at[a, 'total_length']) &
                            (x_test['flags'] == x_sdn.at[a, 'flags']) &
                            (x_test['offset'] == x_sdn.at[a, 'offset']) &
                            (x_test['ttl'] == x_sdn.at[a, 'ttl']) &
                            (x_test['proto'] == x_sdn.at[a, 'proto']) &
                            (x_test['csum'] == x_sdn.at[a, 'csum']) &
                            (x_test['src_ip'] == x_sdn.at[a, 'src_ip']) &
                            (x_test['dst_ip'] == x_sdn.at[a, 'dst_ip']) &
                            (x_test['src_port'] == x_sdn.at[a, 'src_port']) &
                            (x_test['dst_port'] == x_sdn.at[a, 'dst_port']) &
                            (x_test['tcp_flag'] == x_sdn.at[a, 'tcp_flag']) &
                            (x_test['type_icmp'] == x_sdn.at[a, 'type_icmp']) &
                            (x_test['code_icmp'] == x_sdn.at[a, 'code_icmp']) &
                            (x_test['csum_icmp'] == x_sdn.at[a, 'csum_icmp']) &
                            (x_test['port_no'] == x_sdn.at[a, 'port_no'])].index.tolist()

        if len(index_list) == 1:
            datatemp = np.append(datatemp, int(a))
            data = np.append(data, int(index_list[0]))
        else:
            i = i + 1
            print(i)

    np.save(output, data)
    np.save(outputs, datatemp)

getindex('ensemble_boosting.csv', 'indexreal.npy', 'indexsims.npy')

