import pandas as pd

# Membaca dataset .csv
df = pd.read_csv('TEST-DATA.csv')

# Memeriksa label yang ada dalam dataset
print(df['label'].value_counts())  # Asumsi kolom 'label' adalah nama kolom label

# Downsampling: mengambil 3.000 data secara acak dari setiap label
df_downsampled = df.groupby('label').apply(lambda x: x.sample(n=3000, random_state=42)).reset_index(drop=True)

# Memeriksa kembali distribusi data setelah downsampling
print(df_downsampled['label'].value_counts())

# Menyimpan dataset hasil downsampling
df_downsampled.to_csv('18000.csv', index=False)

