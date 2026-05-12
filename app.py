"""
Al-Mizan Online Quran Academy — AI Chatbot
==========================================
RAG-powered chatbot using Groq API + FAISS + Sentence Transformers
Answers student queries about courses, pricing, scheduling, and more.

Author: Ubaid ur Rehman
GitHub: github.com/Ub207
"""

import streamlit as st
import os
import numpy as np
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Al-Mizan Quran Academy — AI Assistant",
    page_icon="🕌",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS — Light Warm Islamic Theme ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Poppins:wght@300;400;500;600&display=swap');

/* ── Background ── */
.stApp {
    background: #fdfaf5;
    font-family: 'Poppins', sans-serif;
}

/* hide default streamlit top bar */
header[data-testid="stHeader"] { background: transparent; }

/* ── Header ── */
.academy-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    background: linear-gradient(180deg, #fff8ee 0%, #fdfaf5 100%);
    border-bottom: 2px solid #e8d8b0;
    margin-bottom: 1.5rem;
    border-radius: 0 0 16px 16px;
}
.academy-header .bismillah {
    font-family: 'Amiri', serif;
    color: #b8860b;
    font-size: 1.8rem;
    margin-bottom: 6px;
}
.academy-header h1 {
    font-family: 'Amiri', serif;
    color: #3a2a00;
    font-size: 1.9rem;
    margin-bottom: 6px;
}
.academy-header p {
    color: #7a6a50;
    font-size: 0.9rem;
    font-weight: 300;
}

/* ── Chat area background ── */
[data-testid="stChatMessageContainer"],
.stChatFloatingInputContainer {
    background: #fdfaf5;
}

/* ── Chat messages shared ── */
.stChatMessage {
    font-family: 'Poppins', sans-serif !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07) !important;
}

/* All text inside messages */
.stChatMessage p,
.stChatMessage li,
.stChatMessage ol,
.stChatMessage ul,
.stChatMessage strong,
.stChatMessage em,
.stChatMessage span,
.stChatMessage [data-testid="stMarkdownContainer"],
.stChatMessage [data-testid="stMarkdownContainer"] * {
    color: #2a1f00 !important;
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
}

/* ── User bubble — warm gold tint ── */
[data-testid="stChatMessageContent"]:has(~ [data-testid="chatAvatarIcon-user"]),
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, #fff8e1, #fef3c7) !important;
    border: 1px solid #d4a017 !important;
}

/* ── Assistant bubble — clean white ── */
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #ffffff !important;
    border: 1px solid #e8dcc8 !important;
}

/* ── Quick question buttons ── */
.stButton > button {
    background: #ffffff !important;
    color: #8a6d1e !important;
    border: 1.5px solid #c8a96e !important;
    border-radius: 22px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #c8a96e !important;
    color: #ffffff !important;
    border-color: #c8a96e !important;
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(200,169,110,0.35) !important;
}

/* Clear Chat button — red tint */
.stButton > button[kind="secondary"] {
    border-color: #e57373 !important;
    color: #c62828 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #e57373 !important;
    color: #fff !important;
}

/* ── Chat input box ── */
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #2a1f00 !important;
    border: 1.5px solid #d4b86a !important;
    border-radius: 12px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #b0956a !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #fff8ee !important;
    border-right: 1px solid #e8d8b0;
}
section[data-testid="stSidebar"] * {
    color: #3a2a00 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #e8d8b0;
}

/* ── Section header (Quick Questions label) ── */
h4 { color: #5a4010 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1rem;
    color: #a08050;
    font-family: 'Poppins', sans-serif;
    font-size: 0.75rem;
    margin-top: 2rem;
    border-top: 1px solid #e8d8b0;
}
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_loaded" not in st.session_state:
    st.session_state.kb_loaded = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None


# --- Load Knowledge Base & Build Index ---
@st.cache_resource(show_spinner="Loading Academy Knowledge Base...")
def load_knowledge_base():
    """Load KB, chunk it, embed it, build FAISS index."""
    from sentence_transformers import SentenceTransformer
    import faiss

    # Read knowledge base
    kb_path = Path(__file__).parent / "quran_academy_kb.txt"
    if not kb_path.exists():
        st.error("Knowledge base file not found! Please add quran_academy_kb.txt")
        st.stop()

    text = kb_path.read_text(encoding="utf-8")

    # Split into chunks (by Q&A pairs)
    raw_chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if line.startswith("Q:") and current_chunk:
            raw_chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk.strip():
        raw_chunks.append(current_chunk.strip())

    # Filter out section headers (chunks without Q:)
    chunks = [c for c in raw_chunks if "Q:" in c or "A:" in c]

    # If no proper Q&A chunks found, fall back to paragraph splitting
    if not chunks:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

    # Embed chunks
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return chunks, index, embedder


def retrieve_context(query: str, chunks, index, embedder, top_k: int = 3) -> str:
    """Retrieve most relevant chunks for a query."""
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    relevant = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            relevant.append(chunks[idx])

    return "\n\n---\n\n".join(relevant)


def get_groq_response(user_message: str, context: str, chat_history: list) -> str:
    """Get response from Groq API with RAG context."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Check Streamlit secrets
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            return "⚠️ Groq API key not found. Please set GROQ_API_KEY in environment variables or Streamlit secrets."

    client = Groq(api_key=api_key)

    system_prompt = f"""You are the AI Assistant for Al-Mizan Online Quran Academy. Your name is "Al-Mizan Assistant".

ROLE: You help prospective and current students with questions about the academy's courses, pricing, scheduling, enrollment, and Islamic education topics.

PERSONALITY:
- Warm, welcoming, and professional
- Use Islamic greetings naturally (Assalamu Alaikum, InshaAllah, MashaAllah)
- Be encouraging to new learners
- Show respect for the sacred nature of Quran education

KNOWLEDGE BASE CONTEXT:
{context}

RULES:
1. Answer ONLY based on the provided context. If the answer is not in the context, politely say you don't have that specific information and suggest contacting the academy directly.
2. Always be accurate about pricing, course details, and teacher credentials.
3. For enrollment inquiries, guide them to book a FREE trial class.
4. Keep responses concise but helpful (2-4 paragraphs max).
5. If asked about Islamic rulings (Fatwa), politely redirect them to qualified scholars and clarify you only assist with academy-related queries.
6. You can answer basic questions about Tajweed, Qiraat, and Hifz concepts as educational information.
7. End responses with a helpful follow-up suggestion when appropriate.

CONTACT INFO (use when relevant):
- Email: usmanubaidurrehman@gmail.com
- Website: quranconnectacademy.netlify.app
- Free trial class available via WhatsApp"""

    # Build messages for API
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent chat history (last 6 messages for context)
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Sorry, I encountered an error: {str(e)}. Please try again."


# --- Header ---
st.markdown("""
<div class="academy-header">
    <div class="bismillah">بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ</div>
    <h1>🕌 Al-Mizan Online Quran Academy</h1>
    <p>AI-Powered Student Assistant — Ask anything about our courses, pricing & enrollment</p>
</div>
""", unsafe_allow_html=True)


# --- Load KB ---
try:
    chunks, index, embedder = load_knowledge_base()
    st.session_state.kb_loaded = True
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.embedder = embedder
except Exception as e:
    st.error(f"Error loading knowledge base: {e}")
    st.stop()


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🕌 Al-Mizan Academy")
    st.markdown("---")
    st.markdown("**📚 Our Courses:**")
    st.markdown("""
    - Nazra Quran (Basic Reading)
    - Tajweed (Proper Recitation)
    - Hifz-ul-Quran (Memorization)
    - Saba Qiraat (7 Recitation Styles)
    - Islamic Studies
    - Arabic Language
    - Ijazah Program
    """)
    st.markdown("---")
    st.markdown("**💰 Packages (USD/month):**")
    st.markdown("""
    - Starter: $75 (4 classes)
    - Standard: $140 (8 classes)
    - Intensive: $200 (12 classes)
    """)
    st.markdown("---")
    st.markdown("**📞 Contact:**")
    st.markdown("✉️ usmanubaidurrehman@gmail.com")
    st.markdown("🌐 [Visit Website](https://quranconnectacademy.netlify.app)")
    st.markdown("---")
    st.markdown("**🎓 FREE Trial Class Available!**")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Quick Question Buttons ---
if not st.session_state.messages:
    st.markdown("#### 💡 Quick Questions:")
    cols = st.columns(2)
    quick_questions = [
        "What courses do you offer?",
        "What are the pricing packages?",
        "Do you offer a free trial?",
        "What is Saba Qiraat?",
        "Can adults join classes?",
        "How are classes conducted?",
    ]
    for i, q in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()


# --- Display Chat History ---
for message in st.session_state.messages:
    avatar = "🕌" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# --- Chat Input ---
if prompt := st.chat_input("Assalamu Alaikum! Ask me anything about Al-Mizan Academy..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Retrieve context & generate response
    with st.chat_message("assistant", avatar="🕌"):
        with st.spinner("Thinking..."):
            context = retrieve_context(
                prompt,
                st.session_state.chunks,
                st.session_state.index,
                st.session_state.embedder,
            )
            response = get_groq_response(
                prompt, context, st.session_state.messages[:-1]
            )
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# --- Footer ---
st.markdown("""
<div class="footer">
    Powered by Al-Mizan Online Quran Academy | Built with ❤️ using Streamlit & Groq AI<br>
    © 2026 Al-Mizan Online Quran Academy — All Rights Reserved
</div>
""", unsafe_allow_html=True)
