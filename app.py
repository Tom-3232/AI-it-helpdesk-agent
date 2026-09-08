import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
except Exception:
    pass

import streamlit as st
import datetime
from src.config import check_ollama_status
from src.database import get_ticket_stats, get_ticket
from src.rag import get_indexed_chunk_count
from src.tools import create_ticket, detect_priority
from src.graph import run_helpdesk_pipeline
from src.memory import retrieve_user_fact

# Page configuration
st.set_page_config(
    page_title="AI IT Helpdesk Agent",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high contrast, glassmorphic styling, badges, and cards
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Streamlit Chat & Markdown High Contrast Visibility */
    [data-testid="stChatMessage"] {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }
    
    [data-testid="stChatMessageContent"],
    [data-testid="stMarkdownContainer"],
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span,
    .stMarkdown div,
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessageContent"] div,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] div {
        color: #f8fafc !important;
        font-size: 1.05rem !important;
        font-weight: 500 !important;
        line-height: 1.65 !important;
    }
    
    [data-testid="stChatMessageContent"] strong, 
    [data-testid="stChatMessageContent"] b,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b,
    .stMarkdown strong,
    .stMarkdown b {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stChatMessageContent"] h1, 
    [data-testid="stChatMessageContent"] h2, 
    [data-testid="stChatMessageContent"] h3, 
    [data-testid="stChatMessageContent"] h4,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        margin-top: 12px !important;
        margin-bottom: 8px !important;
    }

    [data-testid="stChatMessage"] code {
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Header Banner */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #38bdf8;
        font-size: 2.2rem;
        margin: 0 0 8px 0;
        font-weight: 700;
    }
    .main-header p {
        color: #cbd5e1;
        margin: 0;
        font-size: 1.05rem;
    }
    
    /* Status Badges */
    .status-badge-online {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-badge-offline {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Ticket Priority Badges */
    .priority-high {
        background-color: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .priority-medium {
        background-color: rgba(249, 115, 22, 0.25);
        color: #fdba74;
        border: 1px solid #f97316;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .priority-low {
        background-color: rgba(34, 197, 94, 0.25);
        color: #86efac;
        border: 1px solid #22c55e;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    
    /* Ticket Card Styling */
    .ticket-card {
        background-color: #1e293b;
        border-left: 5px solid #38bdf8;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin: 14px 0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        color: #f1f5f9;
    }
    .ticket-card h3, .ticket-card h4 {
        color: #38bdf8 !important;
        margin-top: 0 !important;
    }
    .ticket-card p {
        color: #e2e8f0 !important;
    }
    .ticket-card.high {
        border-left: 6px solid #ef4444;
    }
    .ticket-card.medium {
        border-left: 6px solid #f97316;
    }
    .ticket-card.low {
        border-left: 6px solid #22c55e;
    }
    
    /* Sidebar Styling & Text Contrast */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }

    /* Sidebar Buttons High-Contrast Styling */
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 8px 12px !important;
        margin-bottom: 4px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    section[data-testid="stSidebar"] button:hover,
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        border-color: #38bdf8 !important;
    }

    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] .stButton > button p,
    section[data-testid="stSidebar"] .stButton > button span {
        color: inherit !important;
        font-weight: 600 !important;
    }
    
    /* Metrics Box */
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-box .val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-box .lbl {
        font-size: 0.8rem;
        color: #cbd5e1;
        text-transform: uppercase;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_ticket" not in st.session_state:
    st.session_state.pending_ticket = None

if "user_name" not in st.session_state:
    # Check long-term memory for saved name
    saved_name = retrieve_user_fact("user_name")
    st.session_state.user_name = saved_name if saved_name else "Throna"

# Check Ollama Status Live
ollama_info = check_ollama_status()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("<h2 style='color:#38bdf8; margin-top: 0; margin-bottom: 10px;'>🎧 IT Control Center</h2>", unsafe_allow_html=True)
    
    # Live System Status
    st.subheader("System Health")
    if ollama_info["online"]:
        if ollama_info["has_model"]:
            st.markdown('<div class="status-badge-online">🟢 Ollama: llama3.2 Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge-offline">🟡 Ollama: Model Missing</div>', unsafe_allow_html=True)
            st.caption(ollama_info["message"])
    else:
        st.markdown('<div class="status-badge-offline">🔴 Ollama: Offline</div>', unsafe_allow_html=True)
        st.warning(ollama_info["message"])

    st.markdown("---")
    
    # New Chat Button
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_ticket = None
        st.rerun()

    # Suggested Questions Buttons
    st.subheader("Quick Help Requests")
    suggested_queries = [
        "My WiFi is not connecting",
        "My laptop is very slow",
        "Outlook keeps crashing",
        "I forgot my password",
        "My printer is not printing",
        "My VPN is not connecting",
        "Create a ticket because my laptop keeps restarting"
    ]
    
    selected_query = None
    for q in suggested_queries:
        if st.button(f"📌 {q}", key=f"btn_{q}", use_container_width=True):
            selected_query = q

    st.markdown("---")

    # Knowledge Base Status
    st.subheader("Knowledge Base")
    chunk_cnt = get_indexed_chunk_count()
    st.info(f"📚 **{chunk_cnt} Chunks** indexed in ChromaDB (sentence-transformers/all-MiniLM-L6-v2)")

    # Ticket Statistics
    st.subheader("Support Metrics")
    stats = get_ticket_stats()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="metric-box"><div class="val">{stats["OPEN"]}</div><div class="lbl">Open Tickets</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="metric-box"><div class="val">{stats["CLOSED"]}</div><div class="lbl">Closed Tickets</div></div>', unsafe_allow_html=True)

    # Interactive Open Ticket Management Section
    st.markdown("---")
    st.subheader("Ticket Admin")
    from src.database import get_all_tickets, close_ticket as db_close_ticket
    with st.expander("📋 View & Close Open Tickets", expanded=False):
        open_tkts = get_all_tickets(status="OPEN")
        if not open_tkts:
            st.caption("🟢 No open tickets in database.")
        else:
            for ot in open_tkts:
                st.markdown(f"**ID:** `{ot['ticket_id']}` | Priority: `{ot['priority']}`\n\n**User:** {ot['user_name']}\n\n**Issue:** {ot['issue']}")
                if st.button(f"✅ Close Ticket {ot['ticket_id']}", key=f"ui_close_{ot['ticket_id']}", use_container_width=True):
                    db_close_ticket(ot['ticket_id'])
                    st.toast(f"Ticket {ot['ticket_id']} successfully CLOSED!")
                    st.rerun()
                st.markdown("---")

# ----------------- MAIN UI -----------------
st.markdown("""
<div class="main-header">
    <h1>🖥️ AI IT Helpdesk Agent</h1>
    <p>Autonomous Agentic AI Support Powered by LangGraph, ChromaDB RAG & SQLite Memory</p>
</div>
""", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        # Render ticket cards if embedded in metadata
        if "ticket_card" in msg and msg["ticket_card"]:
            tkt = msg["ticket_card"]
            p_class = f"priority-{tkt['priority'].lower()}"
            st.markdown(f"""
            <div class="ticket-card {tkt['priority'].lower()}">
                <h4>🎫 Support Ticket Generated</h4>
                <p><strong>Ticket ID:</strong> <code>{tkt['ticket_id']}</code></p>
                <p><strong>User:</strong> {tkt['user_name']}</p>
                <p><strong>Issue:</strong> {tkt['issue']}</p>
                <p><strong>Priority:</strong> <span class="{p_class}">{tkt['priority']}</span></p>
                <p><strong>Status:</strong> <code>{tkt['status']}</code> | <strong>Created:</strong> {tkt['created_at']}</p>
            </div>
            """, unsafe_allow_html=True)

# Handle Human-in-the-Loop Confirmation UI if pending ticket exists
if st.session_state.pending_ticket is not None:
    pending = st.session_state.pending_ticket
    p_badge = f"priority-{pending['priority'].lower()}"
    
    with st.container():
        st.warning("⚠️ **Human-in-the-Loop Confirmation Required**")
        st.markdown(f"""
        <div class="ticket-card {pending['priority'].lower()}">
            <h3>⚠️ Ticket Creation Confirmation</h3>
            <p>I detected a <strong>{pending['priority']}</strong> priority issue requiring IT attention.</p>
            <p><strong>Issue:</strong> {pending['issue']}</p>
            <p><strong>Priority Badge:</strong> <span class="{p_badge}">{pending['priority']}</span></p>
            <p>Would you like me to submit this ticket into the SQLite database?</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_yes, col_no = st.columns([1, 1])
        with col_yes:
            if st.button("✅ YES – Create Ticket", key="hitl_yes", type="primary", use_container_width=True):
                # Call create_ticket tool
                new_tkt = create_ticket(
                    user_name=pending["user_name"],
                    issue=pending["issue"],
                    priority=pending["priority"]
                )
                
                success_text = f"✅ **Ticket successfully created!**\nYour support ticket has been recorded with ID `{new_tkt['ticket_id']}`."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": success_text,
                    "ticket_card": new_tkt
                })
                st.session_state.pending_ticket = None
                st.rerun()
                
        with col_no:
            if st.button("❌ NO – Cancel", key="hitl_no", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🚫 Ticket creation request cancelled gracefully."
                })
                st.session_state.pending_ticket = None
                st.rerun()

# User Input Processing
user_input = st.chat_input("Type your IT query or ticket request...")

# Override input if sidebar button clicked
if selected_query:
    user_input = selected_query

if user_input and st.session_state.pending_ticket is None:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Processing via LangGraph Agent Workflow..."):
        # Run pipeline
        res_state = run_helpdesk_pipeline(
            query=user_input,
            user_name=st.session_state.user_name,
            history=st.session_state.messages[:-1]
        )
        
        # Check if pipeline output updated user name
        if res_state.get("user_name"):
            st.session_state.user_name = res_state["user_name"]
            
        # Check if pending ticket generated
        if res_state.get("pending_ticket"):
            st.session_state.pending_ticket = res_state["pending_ticket"]
            # Render response text leading up to HITL confirmation
            st.session_state.messages.append({
                "role": "assistant",
                "content": res_state.get("response", "")
            })
        else:
            # Standard agent response
            st.session_state.messages.append({
                "role": "assistant",
                "content": res_state.get("response", "")
            })
            
    st.rerun()
