import pandas as pd

file_path = '/home/skripsi/skripsi/TRAIN-DATA.csv'

dataset = pd.read_csv(file_path)

print(dataset.head())

print(dataset.describe())

# Tampilkan kolom-kolom dataset
print(dataset.columns)

jumlah_data = len(dataset)
print("Jumlah data dalam list:", jumlah_data)

print(dataset.info())

print("--------------------------------------")

file_path = '/home/skripsi/skripsi/18000/dt.csv'

dataset = pd.read_csv(file_path)

print(dataset.head())

print(dataset.describe())

# Tampilkan kolom-kolom dataset
print(dataset.columns)

jumlah_data = len(dataset)
print("Jumlah data dalam list:", jumlah_data)

print(dataset.info())

print("--------------------------------------")



