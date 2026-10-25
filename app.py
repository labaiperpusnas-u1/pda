import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Chatbot Arsip Masjid Syuhada",
    page_icon="🕌",
    layout="centered"
)

# 2. Inisialisasi Klien Groq LLM (Mengambil API Key dari st.secrets)
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    groq_client = None

# 3. Fungsi Load Data dan Model Embedding (Cached)
@st.cache_resource
def load_data_and_model():
    df = pd.read_excel('PDA_Project1.xlsx', sheet_name='Sheet1')
    
    # Gabungkan kolom untuk knowledge base
    df['combined_text'] = (
        "ID Dokumen: " + df['Identifier'].astype(str) + ". " +
        "Judul: " + df['Title'].astype(str) + ". " +
        "Kategori/Subjek: " + df['Subject'].astype(str) + ". " +
        "Deskripsi: " + df['Description'].astype(str) + ". " +
        "Pencipta: " + df['Creator'].astype(str) + ". " +
        "Kontributor: " + df['Contributor'].astype(str) + ". " +
        "Cakupan/Lokasi: " + df['Coverage'].astype(str) + "."
    )
    
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    document_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=False)
    
    return df, model, document_embeddings

with st.spinner("Menyiapkan kecerdasan buatan arsip Masjid Syuhada..."):
    df_docs, embedding_model, doc_embeddings = load_data_and_model()

# 4. Tampilan Antarmuka (UI)
st.title("🕌 Asisten Virtual Arsip Masjid Syuhada")
st.markdown("""
Chatbot ini didukung oleh AI interaktif untuk membantu Anda menjelajahi dokumen **Penyusunan Perencanaan Teknis Rehabilitasi Masjid Syuhada** menggunakan bahasa sehari-hari yang luwes.
""")

# Cek apakah API Key sudah dipasang di Streamlit Secrets
if not groq_client:
    st.error("⚠️ GROQ_API_KEY belum dikonfigurasi! Harap tambahkan API Key di Streamlit Secrets Anda.")

# Inisialisasi Riwayat Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait dokumen perencanaan teknis dan rehabilitasi Masjid Syuhada?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Logika RAG dengan LLM
if user_query := st.chat_input("Tanyakan sesuatu mengenai dokumen Masjid Syuhada..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not groq_client:
            st.error("Harap atur Groq API Key terlebih dahulu.")
        else:
            with st.spinner("Sedang merangkum jawaban dengan AI..."):
                # A. Retrieval (Cari dokumen paling relevan)
                query_vector = embedding_model.encode([user_query])
                similarities = cosine_similarity(query_vector, doc_embeddings)[0]
                top_k_indices = np.argsort(similarities)[::-1][:2] # Ambil 2 dokumen teratas
                
                context_texts = []
                sources_used = set()
                for idx in top_k_indices:
                    if similarities[idx] > 0.20:
                        row = df_docs.iloc[idx]
                        context_texts.append(f"- Judul: {row['Title']} (ID: {row['Identifier']})\n  Deskripsi: {row['Description']}\n  Kontributor: {row['Contributor']}")
                        sources_used.add(row['Identifier'])
                
                retrieved_context_str = "\n\n".join(context_texts) if context_texts else "Tidak ada dokumen spesifik yang ditemukan."

                # B. Prompt Engineering untuk LLM agar Luwes & Natural
                system_prompt = f"""
                Anda adalah asisten virtual yang ramah, profesional, dan cerdas untuk arsip proyek "Penyusunan Perencanaan Teknis Rehabilitasi Masjid Syuhada".
                Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan KONTEKS DOKUMEN di bawah ini dengan gaya bahasa manusia yang natural, luwes, dan mudah dipahami.
                Jangan menyebutkan ID dokumen secara kaku kecuali ditanya, cukup jelaskan isinya secara mengalir. Jika informasi tidak ada dalam konteks, katakan dengan sopan bahwa informasi tersebut tidak tersedia di dalam dokumen arsip.

                KONTEKS DOKUMEN:
                {retrieved_context_str}
                """

                # C. Panggil Groq LLM dengan model universal yang sangat stabil
                chat_completion = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",  # Model standar Groq yang pasti ada dan sangat cepat
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                
                response_text = chat_completion.choices[0].message.content
                
                # Tambahkan referensi sumber di bawah jawaban
                if sources_used:
                    response_text += f"\n\n*(Referensi Dokumen ID: {', '.join(list(sources_used))})*"

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
