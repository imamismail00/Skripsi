from operator import attrgetter
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv6
from ryu.lib.packet import ipv4
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import icmp
from ryu.lib.packet import ether_types
import numpy as np
import pandas as pd
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

filepath1 = "TRAIN-DATA.csv"
filepath2 = "36000.csv"

train = pd.read_csv(filepath1)
test = pd.read_csv(filepath2)

train = train[train.notnull()]
test = test[train.notnull()]

train.head()
test.head()

feat_labels = list(train.columns)
print(feat_labels)

print(len(train))
print(len(test))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
spec = preprocessing.LabelEncoder()

datapath_id = preprocessing.LabelEncoder()
version = preprocessing.LabelEncoder()
header_length = preprocessing.LabelEncoder()
tos = preprocessing.LabelEncoder()
total_length = preprocessing.LabelEncoder()
flags = preprocessing.LabelEncoder()
offset = preprocessing.LabelEncoder()
ttl = preprocessing.LabelEncoder()
proto = preprocessing.LabelEncoder()
csum = preprocessing.LabelEncoder()
src_ip = preprocessing.LabelEncoder()
dst_ip = preprocessing.LabelEncoder()
src_port = preprocessing.LabelEncoder()
dst_port = preprocessing.LabelEncoder()
tcp_flag = preprocessing.LabelEncoder()
type_icmp = preprocessing.LabelEncoder()
code_icmp = preprocessing.LabelEncoder()
csum_icmp = preprocessing.LabelEncoder()
port_no = preprocessing.LabelEncoder()
rx_bytes_ave = preprocessing.LabelEncoder()
rx_error_ave = preprocessing.LabelEncoder()
rx_dropped_ave = preprocessing.LabelEncoder()
tx_bytes_ave = preprocessing.LabelEncoder()
tx_error_ave = preprocessing.LabelEncoder()
tx_dropped_ave = preprocessing.LabelEncoder()

x_train = train.drop(columns=['label'])

x_test = test.drop(columns=['label'])

x_trainnew = train.drop(columns=['label'])

x_testnew = train.drop(columns=['label'])

x_trainnew = x_trainnew.apply(le.fit_transform)
x_testnew = x_testnew.apply(le.fit_transform)

datapath_id.fit(x_test['datapath_id'])
version.fit(x_test['version'])
header_length.fit(x_test['header_length'])
tos.fit(x_test['tos'])
total_length.fit(x_test['total_length'])
flags.fit(x_test['flags'])
offset.fit(x_test['offset'])
ttl.fit(x_test['ttl'])
proto.fit(x_test['proto'])
csum.fit(x_test['csum'])
src_ip.fit(x_test['src_ip'])
dst_ip.fit(x_test['dst_ip'])
src_port.fit(x_test['src_port'])
dst_port.fit(x_test['dst_port'])
tcp_flag.fit(x_test['tcp_flag'])
type_icmp.fit(x_test['type_icmp'])
code_icmp.fit(x_test['code_icmp'])
csum_icmp.fit(x_test['csum_icmp'])
port_no.fit(x_test['port_no'])
rx_bytes_ave.fit(x_test['rx_bytes_ave'])
rx_error_ave.fit(x_test['rx_error_ave'])
rx_dropped_ave.fit(x_test['rx_dropped_ave'])
tx_bytes_ave.fit(x_test['tx_bytes_ave'])
tx_error_ave.fit(x_test['tx_error_ave'])
tx_dropped_ave.fit(x_test['tx_dropped_ave'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_important_train = scaler.fit_transform(x_trainnew)
x_important_test = scaler.transform(x_testnew)

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

        # IPV4
        self.version = ""
        self.header_length = ""
        self.tos = ""
        self.total_length = ""
        self.flags = ""
        self.offset = ""
        self.ttl = ""
        self.proto = ""
        self.csum = ""
        self.src_ip = ""
        self.dst_ip = ""
        self.i = 0

        # TCO & UDP
        self.src_port = "0"
        self.dst_port = "0"
        
        self.tcp_flag = "0"
        self.type_icmp = ""
        self.code_icmp = ""
        self.csum_icmp = ""

        self.filename = "ensemble_boosting2.sav"
        self.ensemble_boosting = pickle.load(open(self.filename, 'rb'))

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install table-miss flow entry
        #
        # We specify NO BUFFER to max_len of the output action due to
        # OVS bug. At this moment, if we specify a lesser number, e.g.,
        # 128, OVS will send Packet-In with invalid buffer_id and
        # truncated packet data. In that case, we cannot output packets
        # correctly.  The bug has been fixed in OVS v2.1.0.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        IPV4 = pkt.get_protocols(ipv4.ipv4)
        IPV6 = pkt.get_protocols(ipv6.ipv6)
        UDP = pkt.get_protocols(udp.udp)
        TCP = pkt.get_protocols(tcp.tcp)
        ICMP = pkt.get_protocols(icmp.icmp)

        if (eth.ethertype == ether_types.ETH_TYPE_LLDP) or (len(IPV6) != 0):
            # ignore lldp packet
            return
        else:
            req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
            datapath.send_msg(req)
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        if len(IPV4) != 0:
            self.version = IPV4[0].version
            self.header_length = IPV4[0].header_length
            self.tos = IPV4[0].tos
            self.total_length = IPV4[0].total_length
            self.flags = IPV4[0].flags
            self.offset = IPV4[0].offset
            self.ttl = IPV4[0].ttl
            self.proto = IPV4[0].proto
            self.csum = IPV4[0].csum
            self.src_ip = IPV4[0].src
            self.dst_ip = IPV4[0].dst

        if len(UDP) != 0:
            self.src_port = UDP[0].src_port
            self.dst_port = UDP[0].dst_port

        if len(TCP) != 0:
            self.src_port = TCP[0].src_port
            self.dst_port = TCP[0].dst_port
            self.tcp_flag = TCP[0].bits

        if len(ICMP) != 0:
            self.type_icmp = ICMP[0].type
            self.code_icmp = ICMP[0].code
            self.csum_icmp = ICMP[0].csum

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        self.logger.info(self.i)
        for stat in sorted(body, key=attrgetter("port_no")):
            if stat.port_no == int(self.in_port):
                f2 = open("ensemble_boosting", "a+")
                data = np.array([[datapath_id.transform([ev.msg.datapath.id])[0],
                                  version.transform([self.version])[0],
                                  header_length.transform([self.header_length])[0],
                                  tos.transform([self.tos])[0],
                                  total_length.transform([self.total_length])[0],
                                  flags.transform([self.flags])[0],
                                  offset.transform([self.offset])[0],
                                  ttl.transform([self.ttl])[0],
                                  proto.transform([self.proto])[0],
                                  csum.transform([self.csum])[0],
                                  src_ip.transform([self.src_ip])[0],
                                  dst_ip.transform([self.dst_ip])[0],
                                  src_port.transform([self.src_port])[0],
                                  dst_port.transform([self.dst_port])[0],
                                  tcp_flag.transform([self.tcp_flag])[0],
                                  type_icmp.transform([self.type_icmp])[0],
                                  code_icmp.transform([self.code_icmp])[0],
                                  csum_icmp.transform([self.csum_icmp])[0],
                                  port_no.transform([stat.port_no])[0],
                                  spec.fit_transform([(stat.rx_bytes / stat.rx_packets)])[0],
                                  
                                  rx_error_ave.transform([(stat.rx_errors / stat.rx_packets)])[0],
                                  rx_dropped_ave.transform([((stat.rx_dropped / stat.rx_packets))])[0],
                                  spec.fit_transform([((stat.tx_bytes / stat.tx_packets))])[0],
                                  tx_error_ave.transform([(stat.tx_errors / stat.tx_packets)])[0],
                                  tx_dropped_ave.transform([(stat.tx_dropped / stat.tx_packets)])[0]]])

                data = scaler.transform(data)
                res2 = self.ensemble_boosting.predict(data)

                self.i = self.i + 1
                f2.write(str(ev.msg.datapath.id) + ";" +
                         str(self.version) + ";" +
                         str(self.header_length) + ";" +
                         str(self.tos) + ";" +
                         str(self.total_length) + ";" +
                         str(self.flags) + ";" +
                         str(self.offset) + ";" +
                         str(self.ttl) + ";" +
                         str(self.proto) + ";" +
                         str(self.csum) + ";" +
                         str(self.src_ip) + ";" +
                         str(self.dst_ip) + ";" +
                         str(self.src_port) + ";" +
                         str(self.dst_port) + ";" +
                         str(self.tcp_flag) + ";" +
                         str(self.type_icmp) + ";" +
                         str(self.code_icmp) + ";" +
                         str(self.csum_icmp) + ";" +
                         str(stat.port_no) + ";" +
                         str(res2[0]))
                f2.write("\n")

              

