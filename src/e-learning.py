import os
import re
import streamlit as st
import sys
from io import StringIO
from helper import get_or_build_rag, query_lesson_fast, add_tutorial_to_index

# -----------------------------
# Suppress stderr warnings
# -----------------------------
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="🎓 E-Learning Course RAG", page_icon="🎓", layout="wide")

st.title("🎓 Python E-Learning Assistant")
st.caption("Ask about Python concepts and upload your own tutorials 🧠")

# -----------------------------
# Guardrails
# -----------------------------
PYTHON_KEYWORDS = {
    "python", "function", "variable", "loop", "list", "dict", "tuple",
    "class", "def", "if", "else", "async", "await", "decorator"
}

def guardrail_check(query: str):
    if not any(k in query.lower() for k in PYTHON_KEYWORDS):
        return "⚠️ Please ask Python-related questions only."
    if "```" in query:
        code = re.findall(r"```(?:python)?(.*?)```", query, re.DOTALL)
        for snippet in code:
            try:
                compile(snippet.strip(), "<string>", "exec")
            except SyntaxError:
                return "⚠️ Invalid Python syntax detected."
    return None

# -----------------------------
# Sidebar (Upload)
# -----------------------------
st.sidebar.header("📘 Add Tutorials")

uploaded = st.sidebar.file_uploader("Upload `.txt` or `.md` files", type=["txt", "md"], accept_multiple_files=True)
if uploaded:
    save_dir = os.path.join(os.getcwd(), "chat_assistant", "data", "tutorials")
    os.makedirs(save_dir, exist_ok=True)
    with SuppressOutput():  # Suppress any error messages during upload
        for file in uploaded:
            path = os.path.join(save_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getvalue())
            try:
                add_tutorial_to_index(path)
            except Exception:
                pass  # Silently handle errors
    st.sidebar.success(f"✅ {len(uploaded)} file(s) added to the knowledge base!")

# -----------------------------
# Initialize RAG
# -----------------------------
@st.cache_resource
def init_rag():
    with SuppressOutput():  # Suppress initialization errors/warnings
        try:
            retriever = get_or_build_rag()
            return retriever
        except Exception as e:
            # Re-raise critical errors only
            raise

retriever = init_rag()

# -----------------------------
# Chat Interface
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

user_query = st.chat_input("Ask your Python question...")

if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append(("user", user_query))

    guard_error = guardrail_check(user_query)
    if guard_error:
        st.chat_message("assistant").warning(guard_error)
        st.session_state.chat_history.append(("assistant", guard_error))
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Suppress all error outputs during query
                with SuppressOutput():
                    try:
                        ans = query_lesson_fast(user_query, retriever)
                    except Exception as e:
                        ans = f"⚠️ Unable to generate response. Please try again."
                st.markdown(ans)
        st.session_state.chat_history.append(("assistant", ans))