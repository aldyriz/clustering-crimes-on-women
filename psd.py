import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder

def show_data():
    st.header('Menu Data')
    st.write('Ini adalah halaman untuk menampilkan data dari Excel.')

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        st.session_state.upload_count = 0

    if st.session_state.upload_count < 3:
        uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"], key=f"file_uploader_{st.session_state.upload_count}")

        if uploaded_file is not None:
            st.session_state.uploaded_files.append(uploaded_file)
            st.session_state.upload_count += 1

            df = pd.read_excel(uploaded_file)
            st.session_state[f'data_{st.session_state.upload_count}'] = df

    else:
        st.write("Anda telah mengunggah maksimal 3 file.")

    for i in range(1, st.session_state.upload_count + 1):
        df = st.session_state.get(f'data_{i}')
        if df is not None:
            st.write(f"Data dari Upload {i}:")
            st.dataframe(df)
            st.write("Informasi statistik ringkas:")
            st.write(df.describe())

def show_perhitungan():
    st.header('Menu Perhitungan')
    st.write('Ini adalah halaman untuk menghitung data menggunakan metode K-Medoids.')

    if st.session_state.uploaded_files:
        uploaded_file = st.session_state.uploaded_files[0]
        df_perhitungan = pd.read_excel(uploaded_file)

        # Normalisasi nama kolom (hapus spasi ekstra dan ubah ke huruf kecil)
        df_perhitungan.columns = df_perhitungan.columns.str.strip().str.lower()

        # Menampilkan data
        st.write("Data Kejahatan pada Wanita di Negara Bagian India:")
        st.dataframe(df_perhitungan)

        # Kolom yang dibutuhkan untuk clustering
        columns_needed = ['rape', 'kidnapping and abduction cases', 'dowry deaths', 
                          'assault against women', 'assault against modesty of women', 
                          'domestic violence', 'women trafficking']
        
        missing_columns = [col for col in columns_needed if col not in df_perhitungan.columns]
        
        if missing_columns:
            st.error(f"Kolom berikut tidak ditemukan dalam data: {', '.join(missing_columns)}")
            return

        # Preprocessing: Encode categorical columns if exist and handle missing values
        if df_perhitungan.isnull().values.any():
            st.warning("Data mengandung nilai NaN. Melakukan penggantian NaN dengan nilai rata-rata kolom.")
            df_perhitungan = df_perhitungan.fillna(df_perhitungan.mean())

        # Pilih jumlah cluster menjadi 3 (tingkat tinggi, sedang, rendah)
        n_clusters = 3

        if st.button('Lakukan Perhitungan K-Medoids'):
            try:
                # K-Medoids clustering
                kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
                labels = kmedoids.fit_predict(df_perhitungan[columns_needed])

                # Menambahkan kolom Cluster
                df_perhitungan['Cluster'] = labels + 1  # Mulai dari 1 untuk cluster, bukan 0
                st.write("Hasil Clustering:")
                st.dataframe(df_perhitungan)

                # Visualisasi hasil clustering (scatter plot)
                plot_clusters(df_perhitungan, kmedoids)

                # Silhouette Score untuk mengukur kualitas clustering
                silhouette_avg = silhouette_score(df_perhitungan[columns_needed], labels)
                davies_bouldin = davies_bouldin_score(df_perhitungan[columns_needed], labels)
                calinski_harabasz = calinski_harabasz_score(df_perhitungan[columns_needed], labels)

                st.write(f"Silhouette Coefficient: {silhouette_avg:.2f}")
                st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
                st.write(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")

                # Menampilkan distribusi jumlah kasus per cluster
                cluster_counts = df_perhitungan['Cluster'].value_counts().sort_index()
                st.write("Jumlah negara bagian per cluster:")
                st.write(cluster_counts)

                # Menampilkan diagram pie untuk distribusi cluster
                plot_pie_chart(cluster_counts)

                # Menghitung rata-rata atribut per cluster
                # cluster_means = df_perhitungan.groupby('Cluster').mean()
                # st.write("Rata-rata atribut per cluster:")
                # st.write(cluster_means)

                numeric_columns = df_perhitungan.select_dtypes(include=[np.number]).columns
                cluster_means = df_perhitungan.groupby('Cluster')[numeric_columns].mean()
                st.write("Rata-rata atribut per cluster:")
                st.write(cluster_means)

                # Menyediakan analisis berdasarkan hasil clustering
                cluster_labels = {1: 'Tingkat Tinggi', 2: 'Tingkat Sedang', 3: 'Tingkat Rendah'}
                for cluster in range(1, 4):
                    st.write(f"Cluster {cluster} ({cluster_labels[cluster]}):")
                    st.write(df_perhitungan[df_perhitungan['Cluster'] == cluster])

            except ValueError as e:
                st.error(f"Terjadi kesalahan saat melakukan clustering: {str(e)}")

        # Menemukan jumlah cluster terbaik secara otomatis dengan Silhouette Score
        if st.button('Cari Jumlah Cluster Terbaik'):
            silhouette_scores = []
            davies_bouldin_scores = []
            calinski_harabasz_scores = []
            K = range(2, 11)  # Mencoba berbagai nilai K (jumlah cluster)
            for k in K:
                kmedoids = KMedoids(n_clusters=k, random_state=0)
                labels = kmedoids.fit_predict(df_perhitungan[columns_needed])
                silhouette_avg = silhouette_score(df_perhitungan[columns_needed], labels)
                davies_bouldin = davies_bouldin_score(df_perhitungan[columns_needed], labels)
                calinski_harabasz = calinski_harabasz_score(df_perhitungan[columns_needed], labels)

                silhouette_scores.append(silhouette_avg)
                davies_bouldin_scores.append(davies_bouldin)
                calinski_harabasz_scores.append(calinski_harabasz)

            best_k = K[np.argmax(silhouette_scores)]
            st.write(f"Jumlah cluster terbaik adalah: {best_k} dengan Silhouette Coefficient: {max(silhouette_scores):.2f}")

            fig, ax = plt.subplots()
            ax.plot(K, silhouette_scores, 'bo-', markersize=8, label="Silhouette Score")
            ax.plot(K, davies_bouldin_scores, 'ro-', markersize=8, label="Davies-Bouldin Index")
            ax.plot(K, calinski_harabasz_scores, 'go-', markersize=8, label="Calinski-Harabasz Index")
            ax.set_xlabel('Jumlah cluster (k)')
            ax.set_ylabel('Nilai Skor')
            ax.set_title('Evaluasi untuk berbagai jumlah cluster')
            ax.legend()
            st.pyplot(fig)

    else:
        st.write("Silakan unggah file Excel terlebih dahulu di menu Data.")

def plot_clusters(df, kmedoids):
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))

    # Mendapatkan label unik (cluster) dan warna untuk masing-masing cluster
    unique_labels = np.unique(kmedoids.labels_)
    colors = sns.color_palette('husl', n_colors=len(unique_labels))

    # Mendefinisikan keterangan tingkat kejahatan berdasarkan cluster
    cluster_labels = {1: 'Tingkat Tinggi', 2: 'Tingkat Sedang', 3: 'Tingkat Rendah'}

    for label, color in zip(unique_labels, colors):
        subset = df[df['Cluster'] == label + 1]  # Adjust untuk memulai label cluster dari 1
        plt.scatter(subset['rape'], subset['kidnapping and abduction cases'], 
                    label=f'Cluster {label+1} ({cluster_labels[label+1]})', 
                    color=color, alpha=0.7)

    # Menambahkan judul, label sumbu, dan legenda
    plt.title('Clustering Kejahatan pada Wanita di Negara Bagian India')
    plt.xlabel('Kasus Pemerkosaan')
    plt.ylabel('Kasus Penculikan dan Penghilangan')
    plt.legend(title="Tingkat Kejahatan")

    # Menampilkan plot
    st.pyplot(plt)

def plot_pie_chart(cluster_counts):
    # Membuat diagram pie untuk distribusi cluster
    fig, ax = plt.subplots()
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, 
           colors=sns.color_palette('Set3', n_colors=len(cluster_counts)))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Menambahkan judul 
    plt.title('Distribusi Cluster Kejahatan pada Wanita di Negara Bagian India')

    # Menampilkan diagram pie
    st.pyplot(fig)

def main():
    st.title('Clustering Kejahatan pada Wanita di Negara Bagian India')
    menu = ['Data', 'Perhitungan']
    choice = st.sidebar.radio('Pilih Menu', menu)

    if choice == 'Data':
        show_data()
    elif choice == 'Perhitungan':
        show_perhitungan()

if __name__ == "__main__":
    main()
