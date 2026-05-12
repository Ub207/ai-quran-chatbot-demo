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

# --- Custom CSS for Islamic Theme ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Poppins:wght@300;400;500;600&display=swap');

/* Main container */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #1a2f4a 50%, #0d2137 100%);
}

/* Header */
.academy-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(180deg, rgba(212,175,55,0.15) 0%, transparent 100%);
    border-bottom: 2px solid rgba(212,175,55,0.3);
    margin-bottom: 1.5rem;
}

.academy-header h1 {
    font-family: 'Amiri', serif;
    color: #d4af37;
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
    text-shadow: 0 2px 10px rgba(212,175,55,0.3);
}

.academy-header .bismillah {
    font-family: 'Amiri', serif;
    color: rgba(212,175,55,0.8);
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
}

.academy-header p {
    font-family: 'Poppins', sans-serif;
    color: #8fa8c8;
    font-size: 0.95rem;
    font-weight: 300;
}

/* Chat messages */
.stChatMessage {
    font-family: 'Poppins', sans-serif;
    border-radius: 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2137 0%, #1a2f4a 100%);
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #c8d8e8;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #d4af37 0%, #b8962e 100%);
    color: #0a1628;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(212,175,55,0.4);
}

/* Input */
.stChatInput {
    font-family: 'Poppins', sans-serif;
}

/* Quick questions */
.quick-q {
    display: inline-block;
    background: rgba(212,175,55,0.1);
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    margin: 0.2rem;
    color: #d4af37;
    font-family: 'Poppins', sans-serif;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
}

.quick-q:hover {
    background: rgba(212,175,55,0.25);
}

/* Footer */
.footer {
    text-align: center;
    padding: 1rem;
    color: #5a7a9a;
    font-family: 'Poppins', sans-serif;
    font-size: 0.75rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(212,175,55,0.15);
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
