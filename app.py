import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Chatbot Arsip Masjid Syuhada",
    page_icon="🕌",
    layout="centered"
)

# 2. Fungsi Load Data dan Model Embedding (Memakai Cache agar Cepat)
@st.cache_resource
def load_data_and_model():
    # Membaca file excel
    df = pd.read_excel('PDA_Project1.xlsx', sheet_name='Sheet1')
    
    # Menggabungkan kolom penting menjadi satu teks konteks (Knowledge Base)
    df['combined_text'] = (
        "ID Dokumen: " + df['Identifier'].astype(str) + ". " +
        "Judul: " + df['Title'].astype(str) + ". " +
        "Kategori/Subjek: " + df['Subject'].astype(str) + ". " +
        "Deskripsi: " + df['Description'].astype(str) + ". " +
        "Pencipta: " + df['Creator'].astype(str) + ". " +
        "Kontributor: " + df['Contributor'].astype(str) + ". " +
        "Cakupan/Lokasi: " + df['Coverage'].astype(str) + "."
    )
    
    # Menggunakan model embedding multibahasa ringan
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Membuat vektor embedding untuk setiap dokumen
    document_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=False)
    
    return df, model, document_embeddings

# Eksekusi pemuatan data
with st.spinner("Memuat sistem kecerdasan buatan dan database arsip Masjid Syuhada... Mohon tunggu sebentar."):
    df_docs, embedding_model, doc_embeddings = load_data_and_model()

# 3. Tampilan Antarmuka (UI) Chatbot
st.title("🕌 Asisten Virtual Arsip Masjid Syuhada")
st.markdown("""
Chatbot ini dirancang untuk membantu Anda mencari dan memahami isi dokumen **Penyusunan Perencanaan Teknis Rehabilitasi Masjid Syuhada** secara interaktif menggunakan bahasa sehari-hari. 
""")

# Inisialisasi riwayat obrolan (chat history)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya adalah asisten virtual arsip Masjid Syuhada. Ada yang bisa saya bantu terkait dokumen perencanaan teknis dan rehabilitasinya?"}
    ]

# Tampilkan pesan obrolan sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Logika RAG (Retrieval & Natural Response Generation)
if user_query := st.chat_input("Tanyakan sesuatu mengenai dokumen Masjid Syuhada..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi relevan dari dokumen..."):
            # A. Proses Retrieval (Pencarian Berbasis Semantik Vektor)
            query_vector = embedding_model.encode([user_query])
            similarities = cosine_similarity(query_vector, doc_embeddings)[0]
            
            # Ambil dokumen teratas yang paling relevan (Top-2 Dokumen)
            top_k_indices = np.argsort(similarities)[::-1][:2]
            
            retrieved_contexts = []
            sources_used = set()
            
            for idx in top_k_indices:
                score = similarities[idx]
                if score > 0.25: 
                    row = df_docs.iloc[idx]
                    retrieved_contexts.append(
                        f"- **{row['Title']}** (ID: `{row['Identifier']}`)\n"
                        f"  *Deskripsi:* {row['Description']}\n"
                        f"  *Pencipta/Kontributor:* {row['Creator']} / {row['Contributor']}\n"
                        f"  *Subjek:* {row['Subject']}"
                    )
                    sources_used.add(row['Identifier'])

            # B. Pembuatan Respon Luwes
            if retrieved_contexts:
                response_text = "Berikut adalah informasi yang saya temukan berdasarkan dokumen arsip:\n\n"
                response_text += "\n\n".join(retrieved_contexts)
                response_text += f"\n\n*(Referensi Dokumen ID: {', '.join(list(sources_used))})*"
            else:
                response_text = (
                    "Maaf, saya tidak menemukan informasi yang spesifik mengenai pertanyaan tersebut di dalam dokumen arsip "
                    "Perencanaan Teknis Rehabilitasi Masjid Syuhada. Coba tanyakan hal lain terkait laporan kegiatan, "
                    "gambar teknis, sejarah, inventarisasi kerusakan, atau panduan pelestariannya."
                )

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
