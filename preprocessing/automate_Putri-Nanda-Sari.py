import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """Memuat data dari file CSV"""
    return pd.read_csv(filepath)

def scale_data(df):
    """Melakukan scaling pada kolom numerik"""
    scaler = StandardScaler()
    # Scaling kolom numerik yang relevan
    df[['Number of Affected Users', 'Incident Resolution Time (in Hours)']] = scaler.fit_transform(df[['Number of Affected Users', 'Incident Resolution Time (in Hours)']])
    return df

def encode_data(df):
    """Melakukan Label Encoding pada kolom kategorikal"""
    label_encoder = LabelEncoder()
    
    # Encoding kolom kategorikal
    df['Attack Type'] = label_encoder.fit_transform(df['Attack Type'])
    df['Country'] = label_encoder.fit_transform(df['Country'])
    df['Target Industry'] = label_encoder.fit_transform(df['Target Industry'])
    df['Attack Source'] = label_encoder.fit_transform(df['Attack Source'])
    df['Security Vulnerability Type'] = label_encoder.fit_transform(df['Security Vulnerability Type'])
    df['Defense Mechanism Used'] = label_encoder.fit_transform(df['Defense Mechanism Used'])  # Kolom target
    
    return df

def preprocess_data(filepath):
    """Fungsi utama untuk melakukan preprocessing dan mengembalikan dataset yang siap digunakan"""
    # 1. Memuat data
    df = load_data(filepath)
    
    # 2. Melakukan scaling pada data numerik
    df = scale_data(df)
    
    # 3. Melakukan encoding pada data kategorikal
    df = encode_data(df)
    
    # Menyimpan dataset yang sudah diproses ke dalam file CSV
    df.to_csv('processed_dataset.csv', index=False)
    
    # Mengembalikan dataframe yang sudah diproses
    return df
