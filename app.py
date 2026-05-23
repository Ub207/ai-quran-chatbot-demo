import streamlit as st
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ======================
# LOAD ENV FIRST
# ======================
load_dotenv()

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Al-Mizan Quran Academy - AI Assistant",
    page_icon="🕌",
)

# ======================
# SESSION STATE
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================
# LOAD KB + FAISS
# ======================
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_knowledge_base():
    from sentence_transformers import SentenceTransformer
    import faiss

    kb_path = Path(__file__).parent / "quran_academy_kb.txt"

    if not kb_path.exists():
        st.error("Knowledge base file missing: quran_academy_kb.txt")
        st.stop()

    text = kb_path.read_text(encoding="utf-8")

    # Split into chunks by double newline
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index, embedder


def retrieve_context(query, chunks, index, embedder, top_k=3):
    q_emb = embedder.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    _, idxs = index.search(q_emb, top_k)
    return "\n\n".join([chunks[i] for i in idxs[0] if i < len(chunks)])


# ======================
# GROQ RESPONSE
# ======================
def get_groq_response(user_message, context, chat_history):
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            return "❌ GROQ API key missing. Add it in .env or Streamlit secrets."

    client = Groq(api_key=api_key)

    system_prompt = f"""You are a helpful AI assistant for Al-Mizan Online Quran Academy, founded by Qari Hafiz Ubaid ur Rehman.

ROLE:
Help visitors learn about the academy, its courses, enrollment process, and Quran education in general.

RULES:
- Be warm, respectful, and professional
- Use simple and clear language
- Answer in the same language the user writes in (Urdu or English)
- If the answer is not in the context below, politely say you don't have that information and suggest contacting the academy directly
- Keep answers concise but helpful
- Begin responses with Islamic greetings when appropriate (e.g., Assalamu Alaikum)

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    for m in chat_history[-6:]:
        messages.append(m)

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ API Error: {str(e)}"


# ======================
# UI
# ======================
st.title("🕌 Al-Mizan Online Quran Academy")
st.caption("AI Assistant — Ask anything about our courses, enrollment, and more")

st.markdown("---")

chunks, index, embedder = load_knowledge_base()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask about courses, enrollment, fees, timings...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    context = retrieve_context(prompt, chunks, index, embedder)
    response = get_groq_response(prompt, context, st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
