# library manipulation dataset
import pandas as pd

# library manipulation array
import numpy as np

# import library streamlit
import streamlit as st

# library data visualization
import requests
import plotly.express as px
import plotly.graph_objects as go

# config web streamlit
st.set_page_config(
  page_title="UTS - Audi Widya Putra",
  page_icon="🅰",
  layout="wide",
  initial_sidebar_state="auto",
  menu_items={
    "Get Help": "https://github.com/AudiWP",
    "Report a bug": "https://github.com/AudiWP",
    "About": "### Copyright 2024 all rights reserved by Audi Widya Putra"
  }
)

st.markdown("## Data Visualisasi Pima Indians Diabetes")
st.info(" ")


#pembacaan dataset
df = pd.read_csv("Pima_diabetes.csv")

#menampilkan dataframe
with st.container():
  col1, col2 = st.columns([0.5, 0.5])
  with col1:
    st.dataframe(data=df)
  with col2:
    # col1, col2 = st.columns([0.5, 0.5])
    # col1.metric(label="Percentage recovery", value="79.19%", delta="0.35%")
    # col2.metric(label="Percentage dead", value="3.45%", delta="0,00%")
    # st.caption("Summary statistics of covid-19, update at 20 October 2020")
    st.markdown(" ")      
    st.markdown(" ")      
    st.markdown(" ")      
    st.markdown(" ")      
    st.markdown(" ")      
    st.markdown(" ")      
    st.markdown("Dataset ini berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuan dari kumpulan data ini adalah untuk memprediksi secara diagnostik apakah pasien menderita diabetes atau tidak, berdasarkan pengukuran diagnostik tertentu yang disertakan dalam kumpulan data. Beberapa batasan ditempatkan pada pemilihan contoh-contoh ini dari basis data yang lebih besar. Secara khusus, semua pasien di sini adalah perempuan berusia minimal 21 tahun yang berasal dari suku Indian Pima.")
    st.markdown("tautan untuk unduh : (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download)")


st.info(" ")

# Membuat diagram garis
# Mengurutkan data berdasarkan BMI dengan glukosa > 150
filtered_df = df[df['Glucose'] > 150]
sorted_df = filtered_df.sort_values(by='Glucose', ascending=True)

fig = px.line(sorted_df, x='Glucose', y='BMI', markers=True, color_discrete_sequence=['red'], title='Visualisasi Hubungan BMI dengan Glukosa > 150')
st.plotly_chart(fig, use_container_width=True)



#scatter plot diagram
with st.container():
  col1, col2 = st.columns([0.5, 0.5])
  with col1:
      fig = px.scatter(df, x='Glucose', y='Age', color='Age',
                      color_continuous_scale=[(0, "rgb(255,0,0)"), (1, "rgb(0,255,0)")],
                      title='Visualisasi Scatter Plot Glukosa pada Umur')
      st.plotly_chart(fig, use_container_width=True)
  with col2:
    fig = px.scatter(df, x='Glucose', y='BMI', color="Age",
                      color_continuous_scale=px.colors.sequential.Viridis,
                      title='Visualisasi Scatter Plot Glukosa pada BMI')
    st.plotly_chart(fig, use_container_width=True)

#diagram batang untuk 5 nilai Glucose tertinggi
with st.container():
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        df_sorted = df.sort_values(by='Glucose', ascending=False).head(5)  # Mengurutkan dan mengambil 5 data teratas
        fig = px.bar(df_sorted, x='BMI', y='Glucose', 
                     color='Glucose', title='5 Glukosa Tertinggi Berdasarkan BMI')
        st.plotly_chart(fig, use_container_width=True)
    with col2 :
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown("Berdasarkan diagram bar ini, dapat diketahui dari 5 glukosa tertingi memiliki BMI di atas 30. dengan glukosa di atas 197. ")

#grup usia per 10 tahun
df['Age_Group'] = pd.cut(df['Age'], bins=range(20, 100, 10), labels=[f"{i}-{i+9}" for i in range(20, 90, 10)])

#Memfilter data Glucose di atas 150
filtered_df = df[df['Glucose'] > 150]

#Menghitung rata-rata Glucose berdasarkan usia
average_glucose = filtered_df.groupby('Age_Group')['Glucose'].mean().reset_index()

#pie chart
with st.container():
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown(" ")      
        st.markdown("Dalam diagram pie ini, dibuat grup pada umur dengan interval 10 tahun. Dapat diketahui bahwa persentase yang memilki glukosa diatas 150 pada umur 20-29 yaitu 19.7%, umur 30-39 yaitu 19.7%, umur 40-49 yaitu 19.7%, umur 50-59 yaitu 20.5%, dan umur 60-69 yaitu 20.3%.")
    with col2:
        fig = px.pie(average_glucose, values='Glucose', names='Age_Group', hole=0.5, title='Pie Chart Rata-Rata Glukosa > 150 per Kategori Usia')
        st.plotly_chart(fig, use_container_width=True)

#boxplot dan histogram untuk Glukosa
with st.container():
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        fig = px.box(df, y='Glucose', title='Distribusi Glukosa pada Data Diabetes Pima Indians dengan Menggunakan Boxplot', color_discrete_sequence=['green'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='Glucose', nbins=30, title='Distribusi Glucose pada Data Diabetes Pima Indians Menggunakan Histogram', color_discrete_sequence=['red'])

        st.plotly_chart(fig, use_container_width=True)

#Menghitung matriks korelasi

numeric_df = df.select_dtypes(include=[np.number])  # Memilih hanya kolom dengan tipe data numerik
correlation_matrix = numeric_df.corr()

#heatmap dari matriks korelasi
fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                labels=dict(x="Variabel", y="Variabel", color="Korelasi"),
                title="Heatmap Korelasi Antar Variabel pada Data Diabetes Pima Indians",
                color_continuous_scale=px.colors.sequential.Viridis,
                width=1500, height=800)
st.plotly_chart(fig, use_container_width=True)



#footer

st.info("Created By. Audi Widya Putra, May 2024 ")
