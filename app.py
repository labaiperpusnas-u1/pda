import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Coba impor pustaka groq secara opsional
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Chatbot Arsip Masjid Syuhada",
    page_icon="🕌",
    layout="centered"
)

# 2. Inisialisasi Klien Groq (Aman dari Error)
groq_client = None
if GROQ_AVAILABLE:
    try:
        if "GROQ_API_KEY" in st.secrets:
            groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception:
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
Chatbot ini dirancang untuk membantu Anda menjelajahi dokumen **Penyusunan Perencanaan Teknis Rehabilitasi Masjid Syuhada** secara interaktif.
""")

# Status Koneksi AI di sidebar / info kecil
if not groq_client:
    st.info("ℹ️ Berjalan dalam mode **Smart Local RAG** (AI Cloud tidak aktif atau kunci belum dikonfigurasi, menggunakan pencarian dokumen cerdas mandiri).")

# Inisialisasi Riwayat Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait dokumen perencanaan teknis dan rehabilitasi Masjid Syuhada?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Logika RAG & LLM (Dengan Fallback Otomatis)
if user_query := st.chat_input("Tanyakan sesuatu mengenai dokumen Masjid Syuhada..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi relevan..."):
            # A. Retrieval (Cari dokumen paling relevan via Semantic Search)
            query_vector = embedding_model.encode([user_query])
            similarities = cosine_similarity(query_vector, doc_embeddings)[0]
            top_k_indices = np.argsort(similarities)[::-1][:2] # Ambil 2 dokumen teratas
            
            context_texts = []
            sources_used = set()
            for idx in top_k_indices:
                if similarities[idx] > 0.20:
                    row = df_docs.iloc[idx]
                    context_texts.append(
                        f"- **{row['Title']}** (ID: `{row['Identifier']}`)\n"
                        f"  *Deskripsi:* {row['Description']}\n"
                        f"  *Pencipta/Kontributor:* {row['Creator']} / {row['Contributor']}\n"
                        f"  *Subjek:* {row['Subject']}"
                    )
                    sources_used.add(row['Identifier'])
            
            response_text = ""
            ai_success = False

            # B. Coba gunakan Groq LLM jika klien aktif
            if groq_client and context_texts:
                try:
                    retrieved_context_str = "\n\n".join(context_texts)
                    system_prompt = f"""
                    Anda adalah asisten virtual yang ramah dan profesional untuk arsip proyek "Penyusunan Perencanaan Teknis Rehabilitasi Masjid Syuhada".
                    Jawablah pertanyaan pengguna berdasarkan KONTEKS DOKUMEN di bawah ini dengan gaya bahasa manusia yang natural, luwes, dan mudah dipahami.

                    KONTEKS DOKUMEN:
                    {retrieved_context_str}
                    """
                    
                    chat_completion = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        temperature=0.7,
                        max_tokens=1024,
                    )
                    response_text = chat_completion.choices[0].message.content
                    if sources_used:
                        response_text += f"\n\n*(Referensi Dokumen ID: {', '.join(list(sources_used))})*"
                    ai_success = True
                except Exception:
                    ai_success = False # Jika gagal, otomatis turun ke mode lokal

            # C. Fallback: Jika Groq gagal/tidak aktif, tampilkan hasil penarikan dokumen secara rapi
            if not ai_success:
                if context_texts:
                    response_text = "Berikut adalah informasi yang saya temukan berdasarkan arsip dokumen:\n\n"
                    response_text += "\n\n".join(context_texts)
                    response_text += f"\n\n*(Referensi Dokumen ID: {', '.join(list(sources_used))})*"
                else:
                    response_text = (
                        "Maaf, saya tidak menemukan informasi yang spesifik mengenai pertanyaan tersebut di dalam dokumen arsip "
                        "Perencanaan Teknis Rehabilitasi Masjid Syuhada. Coba tanyakan hal lain terkait laporan kegiatan, "
                        "gambar teknis, sejarah, inventarisasi kerusakan, atau panduan pelestariannya."
                    )

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
