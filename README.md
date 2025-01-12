# Deteksi Serangan LRDDoS Menggunakan Ensemble Boosting dengan DecessionTree dan Adaboost

## Deskripsi Dataset 

Dataset yang digunakan memiliki fitur :
1	datapath_id
2	version
3	header_length
4	tos
5	total_length
6	flags
7	offset
8	ttl
9	proto
10	csum
11	src_ip
12	dst_ip
13	src_port
14	dst_port
15	tcp_flag
16	type_icmp
17	code_icmp
18	csum_icmp
19	port_no
20	rx_bytes_ave
21	rx_error_ave
22	rx_dropped_ave
23	tx_bytes_ave
24	tx_error_ave
25	tx_dropped_ave
Seluruh fitur dalam tabel 3.3 digunakan untuk proses training model klasifikasi dan menjadi parameter input saat model diimplementasikan kedalam sistem deteksi. Dataset ini berjumlah 600.000 data yang terdiri dari 420.000 data train dan 180.000 data test. Pada penelitian ini, data test yang akan digunakan sebesar 10% dan 20% yang berjumlah 18.000 dan 36.000 data. Untuk menyeleksi data test tersebut, penulis menggunakan metode random sampling.

### Teknik Deep Learning yang digunakan

* Model dengan menggunakan algoritma Decision Tree
* Model dengan menggunakan algoritma Adaboost

## Authors

Kontributor dalam projek ini yaitu :
* Imam Ismail Tambili, 202010370311403 
