#!/usr/bin/env python3
"""
Streamlit RAG Chatbot UI for Turkish Medical LLM - Professional Version
"""
import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import sys

# Add src to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.rag.r_2c_rag_chain import TurkishMedRAG
from src.utils.logger import setup_logger

logger = setup_logger("StreamlitRAG")

# Page configuration
st.set_page_config(
    page_title="Turkish Medical RAG - Professional Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
    :root {
        --primary-blue: #1e40af;
        --teal: #0f766e;
        --light-blue: #0284c7;
    }
    
    * {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    .main {
        padding: 2rem 1.5rem;
        background-color: #f9fafb;
    }
    
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #0284c7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fef08a 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #f59e0b;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #dcfce7 0%, #ccfbf1 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #10b981;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .header-title {
        background: linear-gradient(135deg, #1e40af 0%, #0f766e 100%);
        padding: 2.5rem 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .header-title h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .header-title p {
        margin: 0.25rem 0 0 0;
        font-size: 1.1em;
        opacity: 0.95;
    }
    
    .source-item {
        background-color: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.75rem 0;
        border-left: 3px solid #0284c7;
        font-size: 0.95em;
    }
    
    .footer-section {
        background: linear-gradient(135deg, #1e40af 0%, #0f766e 100%);
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        margin-top: 3rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        color: white;
    }
    
    .footer-section h3 {
        font-size: 1.5em;
        margin: 0 0 1.5rem 0;
        font-weight: 700;
        text-align: center;
    }
    
    .footer-tech-stack {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .footer-tech-stack p {
        margin: 0.75rem 0;
        font-size: 0.95em;
        line-height: 1.6;
        color: #e0f2fe;
    }
    
    .footer-disclaimer {
        background: rgba(255,215,0,0.1);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #fbbf24;
        margin: 1.5rem 0;
    }
    
    .footer-disclaimer p {
        margin: 0.5rem 0;
        font-size: 0.9em;
        line-height: 1.7;
        color: #fef3c7;
    }
    
    .footer-disclaimer strong {
        color: #fef08a;
    }
    
    .footer-copyright {
        text-align: center;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.2);
        font-size: 0.85em;
        color: #cbd5e1;
        margin-top: 1.5rem;
    }
    
    /* Minimal sidebar spacing */
    [data-testid="stSidebar"] {
        gap: 0.25rem;
    }
    
    [data-testid="stSidebar"] > div > div:first-child {
        gap: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
    st.session_state.rag_initialized = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_queries': 0,
        'total_tokens': 0,
        'start_time': datetime.now()
    }

# Sidebar - Professional Panel
with st.sidebar:
    st.markdown("<div class='header-title'><h3>🏥 Medical RAG Assistant</h3></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    
    # System Configuration
    with st.expander("⚙️ System Configuration", expanded=False):
        st.markdown("""
        **AI Model Stack:**
        - 🧠 Embedding Model: bge-m3 (1024D)
        - 💬 LLM: Qwen 3-8B
        - 🗄️ Vector Store: ChromaDB
        - 🔗 Framework: LangChain
        - 🛠️ Interface: Streamlit
        
        **Features:**
        - Retrieval-Augmented Generation (RAG)
        - Context-aware medical responses
        - Source document tracking
        - Turkish language support
        """)
    
    # Quick Templates
    st.markdown("**💡 Quick Templates**")
    example_queries = [
        "Diyabet belirtileri nelerdir?",
        "Hipertansiyon tedavisi nasıl yapılır?",
        "Kalp krizi acil müdahalesi",
        "Astım atağında ilk yardım",
        "Antibiyotik yan etkileri"
    ]
    
    selected_example = st.selectbox(
        "Select template:",
        example_queries,
        key="example_selector",
        label_visibility="collapsed"
    )
    
    if st.button("📋 Use Template", use_container_width=True):
        st.session_state.query_input = selected_example
        st.rerun()

# Professional Header
st.markdown("""
    <div class="header-title">
    <h1>🏥 Turkish Medical Information Assistant</h1>
    <p>AI-Powered Medical Knowledge Retrieval System</p>
    </div>
""", unsafe_allow_html=True)

# Professional Legal Notice
with st.container():
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Important Notice</h4>
    <p><strong>Educational Use Only:</strong> This system provides medical information for educational purposes and is powered by AI. 
    It is <strong>NOT</strong> a substitute for professional medical consultation. Always consult qualified healthcare professionals 
    for diagnosis, treatment decisions, and medical emergencies.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Auto-initialize RAG if not already done
if not st.session_state.rag_initialized and st.session_state.rag is None:
    try:
        st.session_state.rag = TurkishMedRAG()
        st.session_state.rag_initialized = True
        logger.info("RAG system auto-initialized")
    except Exception as e:
        logger.error(f"Auto-initialization failed: {e}")

# Main Content - Conversation Display
if len(st.session_state.conversation_history) > 0:
    st.markdown("### 📝 Conversation History")
    conversation_container = st.container()
    
    with conversation_container:
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👨‍⚕️"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg["content"])
                    
                    # Show sources
                    if "sources" in msg and msg["sources"]:
                        with st.expander("📚 Information Sources"):
                            for j, source in enumerate(msg["sources"], 1):
                                if source:
                                    st.markdown(f"<div class='source-item'><strong>Source {j}:</strong> {source}</div>", unsafe_allow_html=True)
    
    st.markdown("---")

# Input Section - Always Visible
st.markdown("### ❓ Ask Your Medical Question")

col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "Enter your medical question:",
        placeholder="Example: Diyabet nedir? Hipertansiyon tedavisi nasıl yapılır?",
        key="query_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("Submit", use_container_width=True, type="primary", key="send_btn")

# Process Query
if send_button and user_query:
    if not st.session_state.rag_initialized:
        with st.spinner("⏳ Initializing system..."):
            try:
                st.session_state.rag = TurkishMedRAG()
                st.session_state.rag_initialized = True
                logger.info("RAG system initialized")
            except Exception as e:
                st.error(f"Error initializing system: {e}")
                logger.error(f"Initialization failed: {e}")
    
    if st.session_state.rag_initialized:
        with st.spinner("🔄 Retrieving medical information..."):
            try:
                # Add user message
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # Get response from RAG
                result = st.session_state.rag.answer(user_query)
                
                # Safely extract response data
                if isinstance(result, dict):
                    answer = result.get('answer', 'No response generated')
                    sources = result.get('retrieved_titles', [])
                else:
                    answer = str(result)
                    sources = []
                
                # Add assistant message
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources if sources else []
                })
                
                # Update stats
                st.session_state.stats['total_queries'] += 1
                logger.info(f"Query processed: {user_query[:50]}...")
                st.rerun()
                
            except Exception as e:
                error_msg = f"System Error: {str(e)[:200]}"
                st.error(error_msg)
                logger.error(f"Query error: {e}", exc_info=True)

# Export Options
if len(st.session_state.conversation_history) > 0:
    st.markdown("---")
    st.markdown("### 📥 Export Conversation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 JSON", use_container_width=True, key="json_export"):
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "conversation": st.session_state.conversation_history
                }
                st.download_button(
                    label="Download",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"medical_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📄 Text", use_container_width=True, key="txt_export"):
                export_text = "Medical Assistant Conversation\n"
                export_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                export_text += "=" * 60 + "\n\n"
                
                for msg in st.session_state.conversation_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    export_text += f"[{role}]\n{msg['content']}\n\n"
                
                st.download_button(
                    label="Download",
                    data=export_text,
                    file_name=f"medical_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
                st.session_state.conversation_history = []
                st.session_state.stats['total_queries'] = 0
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85em; padding: 1rem 0;'>
All Rights Reserved © 2026 ❤️ <strong>TurkishMedLLM</strong>
</div>
""", unsafe_allow_html=True)


