import os
import time
import tempfile
import streamlit as st
import fitz
from rl_tutor import get_teaching_strategy
from rag_handler import process_and_add_pdf, query_knowledge_base, clear_knowledge_base

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

MODEL_PATH = os.path.join(os.getcwd(), "model.gguf")

st.set_page_config(page_title="Local AI Tutor", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def get_llm():
    if LLAMA_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=max(1, (os.cpu_count() or 4) - 1),
                n_batch=128,
                n_gpu_layers=0,
                verbose=False
            )
            return llm
        except Exception as e:
            st.error(f"Error loading LLM: {e}")
            return None
    return None

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "fatigue" not in st.session_state:
    st.session_state["fatigue"] = 0.0
if "last_message_time" not in st.session_state:
    st.session_state["last_message_time"] = time.time()
if "rag_active" not in st.session_state:
    st.session_state["rag_active"] = False

llm = get_llm()

with st.sidebar:
    st.title("Teaching Dashboard")
    
    st.header("Upload Materials")
    uploaded_file = st.file_uploader("Upload a PDF to learn from...", type=["pdf"])
    
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = []
        
    if uploaded_file is not None and uploaded_file.name not in st.session_state["processed_files"]:
        with st.spinner("Extracting text from PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                chunks_added = process_and_add_pdf(tmp_path)
                os.remove(tmp_path)
                
                if chunks_added > 0:
                    st.session_state["processed_files"].append(uploaded_file.name)
                    st.session_state["rag_active"] = True
                    st.success(f"✅ Indexed {chunks_added} chunks into knowledge base!")
                else:
                    st.error("No readable text found in PDF (might be a scanned image!)")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                
    st.divider()
    
    st.header("Real-time RL Metrics")
    f_val = st.session_state["fatigue"]
    st.progress(f_val, text=f"Estimated Fatigue: {f_val:.2f}/1.0")
    
    if "latest_strategy" in st.session_state:
        st.metric(label="Current Teaching Style", value=st.session_state["latest_strategy"])
    else:
        st.metric(label="Current Teaching Style", value="None (Waiting for Input)")
        
    st.divider()
    if st.button("Start New Chat"):
        st.session_state["messages"] = []
        st.session_state["fatigue"] = 0.0
        st.session_state["rag_active"] = False
        st.session_state["last_message_time"] = time.time()
        clear_knowledge_base()
        if "latest_strategy" in st.session_state:
            del st.session_state["latest_strategy"]
        st.rerun()
        
    st.divider()
    if not LLAMA_AVAILABLE:
        st.warning("`llama-cpp-python` is not installed! Running in Mock Testing Mode.")
    elif not os.path.exists(MODEL_PATH):
        st.warning(f"Model not found at `{MODEL_PATH}`. Running in Mock Testing Mode.")


st.title("Study Planner AI")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your tutor a question..."):
    current_time = time.time()
    typing_delay = current_time - st.session_state["last_message_time"]
    response_length = len(prompt.split())
    fatigue = st.session_state["fatigue"]
    
    action, strategy_name = get_teaching_strategy(typing_delay, response_length, fatigue)
    st.session_state["latest_strategy"] = strategy_name
    
    if action == 2:
        st.session_state["fatigue"] = max(0.0, fatigue - 0.4)
    else:
        st.session_state["fatigue"] = min(1.0, fatigue + 0.05)
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ✅ THIS IS THE FIX - queries ChromaDB instead of using raw pdf_context
    context = query_knowledge_base(prompt) if st.session_state["rag_active"] else ""
    
    system_instruction = f"You are a helpful AI tutor. You MUST adopt the following pedagogical strategy constraint: {strategy_name}."
    if action == 0:
        system_instruction += " Explain the concept using a relatable analogy."
    elif action == 1:
        system_instruction += " Reply with a thought-provoking Socratic question to guide the student instead of giving the direct answer."
    elif action == 2:
        system_instruction += " The student seems fatigued. Be extremely brief, reassuring, and suggest they take a short break before continuing."
        
    final_user_prompt = prompt
    if context:
        final_user_prompt = f"Use the following textbook context to ground your answer and ONLY answer based on this context if it is relevant:\n\n---\n{context}\n---\n\nStudent Question: {prompt}"
        
    with st.chat_message("assistant"):
        ctx_status = "📄 Document Analysis Active" if context else "🗣️ General Knowledge Chat"
        st.caption(f"*{ctx_status}* | *Strategy Applied: {strategy_name}*")
        
        if llm is None:
            mock_response = f"[Mock Mode - System Instructions]\n\n**Strategy Used:** {strategy_name}\n**Delay:** {typing_delay:.1f}s | **Length:** {response_length} words\n**RAG Context Injected:** {bool(context)}"
            st.markdown(mock_response)
            st.session_state.messages.append({"role": "assistant", "content": mock_response})
            st.session_state["last_message_time"] = time.time()
            st.rerun()
        else:
            # Reconstruct context-aware messages array
            chat_messages = [{"role": "system", "content": system_instruction}]
            # Add all historical messages except the last one (which we'll replace with the context-injected prompt)
            for m in st.session_state.messages[:-1]:
                chat_messages.append({"role": m["role"], "content": m["content"]})
            # Add the last message with its full RAG context
            chat_messages.append({"role": "user", "content": final_user_prompt})

            def generate_response():
                stream = llm.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=256,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        yield delta['content']
                    
            response = st.write_stream(generate_response())
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state["last_message_time"] = time.time()
            st.rerun()