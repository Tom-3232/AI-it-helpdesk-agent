import re
from langchain_ollama import ChatOllama
from src.config import check_ollama_status, DEFAULT_MODEL
from src.rag import query_rag
from src.tools import detect_priority, check_ticket_status, create_ticket, get_current_time
from src.memory import extract_and_store_memories, retrieve_user_fact, format_all_user_memories

# Helper to get LLM instance
def get_llm():
    try:
        return ChatOllama(model=DEFAULT_MODEL, temperature=0)
    except Exception as e:
        print(f"[LLM Warning] Could not initialize ChatOllama: {e}")
        return None

def classify_query(query: str) -> str:
    """
    Classifies an incoming user query into one of three categories:
    'technical' | 'ticket' | 'general'
    """
    text = query.lower().strip()
    
    # 1. Ticket patterns
    ticket_triggers = [
        "ticket", "tkt-", "support ticket", "open ticket", "create a ticket",
        "create ticket", "submit ticket", "check ticket", "ticket status"
    ]
    if any(tg in text for tg in ticket_triggers):
        return "ticket"
        
    # High priority issue direct ticket intent: e.g. "Create a ticket because my laptop keeps restarting"
    if "restarting" in text and ("create" in text or "ticket" in text or "restarts" in text):
        return "ticket"

    # 2. General / Memory patterns
    general_triggers = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
        "my name is", "what is my name", "who am i", "remember", "time", "date",
        "what time", "thank you", "thanks", "bye"
    ]
    if any(text.startswith(gt) or gt in text for gt in general_triggers):
        # Exception: if query also mentions specific tech problem like "wifi", prioritize technical unless it's explicitly asking name/greeting
        if not any(tech_kw in text for tech_kw in ["wifi", "printer", "vpn", "outlook", "slow laptop"]):
            return "general"
            
    # 3. Technical patterns
    tech_keywords = [
        "wifi", "wi-fi", "internet", "connecting", "slow", "printer", "printing",
        "software", "crash", "crashing", "password", "forgot", "vpn", "email",
        "outlook", "laptop", "blue screen", "reboot", "network"
    ]
    if any(kw in text for kw in tech_keywords):
        return "technical"

    # Try LLM classification if Ollama is available
    status = check_ollama_status()
    if status["online"] and status["has_model"]:
        try:
            llm = get_llm()
            if llm:
                prompt = (
                    f"Classify the following IT query into exactly one word: 'technical', 'ticket', or 'general'.\n"
                    f"Query: \"{query}\"\n"
                    f"Category:"
                )
                res = llm.invoke(prompt).content.strip().lower()
                for label in ["technical", "ticket", "general"]:
                    if label in res:
                        return label
        except Exception:
            pass

    return "general"

def run_classifier_node(state: dict) -> dict:
    """Classifier Node in LangGraph."""
    query = state.get("messages", [])[-1].get("content", "") if state.get("messages") else ""
    category = classify_query(query)
    state["classification"] = category
    return state

def run_technical_agent_node(state: dict) -> dict:
    """Technical Agent Node utilizing ChromaDB RAG and ChatOllama."""
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else ""
    
    # 1. RAG context retrieval
    rag_res = query_rag(query, top_k=3)
    context = rag_res.get("context", "")
    state["retrieved_context"] = context
    
    # 2. Check Ollama status
    status = check_ollama_status()
    
    if status["online"] and status["has_model"]:
        try:
            llm = get_llm()
            prompt = (
                f"You are an expert IT Helpdesk Support Specialist.\n"
                f"Use the following authoritative troubleshooting documentation to answer the user's issue concisely and clearly with step-by-step instructions.\n\n"
                f"--- KNOWLEDGE BASE CONTEXT ---\n"
                f"{context}\n\n"
                f"--- USER ISSUE ---\n"
                f"{query}\n\n"
                f"Provide a clear, formatted troubleshooting response:"
            )
            response_text = llm.invoke(prompt).content
        except Exception as e:
            response_text = (
                f"🔧 **IT Troubleshooting Guide** (RAG Knowledge Base):\n\n"
                f"{context}\n\n"
                f"*(Note: Generated directly from vector store. LLM response error: {e})*"
            )
    else:
        response_text = (
            f"🔧 **IT Support Troubleshooting Steps** (Retrieved from Knowledge Base):\n\n"
            f"{context}\n\n"
            f"⚠️ *Ollama status: {status['message']}*"
        )

    state["response"] = response_text
    return state

def run_ticket_agent_node(state: dict) -> dict:
    """Ticket Agent Node with Priority Detection and Human-in-the-Loop approval flags."""
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else ""
    user_name = state.get("user_name", "User")
    
    # Check if this is a ticket lookup
    tkt_match = re.search(r"TKT-2026-\d{5}", query, re.IGNORECASE)
    if tkt_match or "check" in query.lower() or "status" in query.lower():
        if tkt_match:
            tkt_id = tkt_match.group(0).upper()
            status_info = check_ticket_status(tkt_id)
            state["response"] = status_info
            state["pending_ticket"] = None
            return state
        elif "status" in query.lower() and not ("create" in query.lower() or "open" in query.lower()):
            state["response"] = "Please provide your Ticket ID (e.g., TKT-2026-12345) to check its current status."
            state["pending_ticket"] = None
            return state

    # Otherwise, it's a request to create a ticket
    priority = detect_priority(query)
    
    # Prepare pending ticket data for Human-in-the-Loop Streamlit confirmation
    pending_ticket = {
        "user_name": user_name,
        "issue": query,
        "priority": priority,
        "status": "PENDING_CONFIRMATION"
    }
    
    confirmation_msg = (
        f"I detected a **{priority}** priority issue.\n\n"
        f"**Issue Description:** {query}\n"
        f"**Detected Priority:** `{priority}`\n\n"
        f"Would you like me to create the IT support ticket in the database?"
    )
    
    state["pending_ticket"] = pending_ticket
    state["response"] = confirmation_msg
    return state

def run_general_agent_node(state: dict) -> dict:
    """General Agent Node handling small talk, time queries, and long-term memory."""
    messages = state.get("messages", [])
    query = messages[-1].get("content", "") if messages else ""
    user_name = state.get("user_name", "User")
    query_lower = query.lower().strip()
    
    # 1. Store memory if user provides information
    learned_name = extract_and_store_memories(query)
    if learned_name:
        state["user_name"] = learned_name
        state["response"] = f"Nice to meet you, **{learned_name}**! I've saved your name in long-term memory."
        return state
        
    # 2. Check if user asks "What is my name?" or "Who am I?"
    if "my name" in query_lower or "who am i" in query_lower:
        stored_name = retrieve_user_fact("user_name")
        if stored_name:
            state["user_name"] = stored_name
            state["response"] = f"Your name is **{stored_name}**, as saved in your long-term memory profile."
        else:
            state["response"] = "I don't have your name saved in long-term memory yet. You can tell me by saying 'My name is [Your Name]'."
        return state

    # 3. Time / Date queries
    if "time" in query_lower or "date" in query_lower or "day" in query_lower:
        state["response"] = get_current_time()
        return state

    # 4. Greetings and general LLM conversation
    status = check_ollama_status()
    if status["online"] and status["has_model"]:
        try:
            llm = get_llm()
            memories_ctx = format_all_user_memories()
            prompt = (
                f"You are a friendly, helpful IT Helpdesk virtual assistant.\n"
                f"User Profile Memory:\n{memories_ctx}\n\n"
                f"User: {query}\n"
                f"Assistant:"
            )
            res = llm.invoke(prompt).content
            state["response"] = res
            return state
        except Exception as e:
            pass

    # Fallback greeting response
    if any(g in query_lower for g in ["hello", "hi", "hey"]):
        state["response"] = f"Hello! How can I assist you with your IT equipment or support tickets today?"
    else:
        state["response"] = f"I'm here to help with IT troubleshooting, password resets, Wi-Fi issues, software repairs, and support tickets."

    return state
