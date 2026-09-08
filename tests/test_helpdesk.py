import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
except Exception:
    pass

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.database import init_db, get_ticket_stats
from src.rag import get_or_create_vectorstore, query_rag
from src.tools import create_ticket, check_ticket_status, detect_priority
from src.graph import run_helpdesk_pipeline
from src.memory import save_user_memory, get_user_memory, retrieve_user_fact

def run_tests():
    print("=========================================")
    print("  RUNNING AI IT HELPDESK SUITE VERIFICATION")
    print("=========================================\n")
    
    # 1. Database Initialization
    print("[1/6] Testing Database Setup...")
    init_db()
    stats = get_ticket_stats()
    print(f"   [OK] Database initialized successfully. Current stats: {stats}")
    
    # 2. Priority Detection
    print("\n[2/6] Testing Priority Detection...")
    p1 = detect_priority("My laptop keeps restarting constantly")
    p2 = detect_priority("My WiFi connection is slow")
    p3 = detect_priority("Need paper for printer")
    assert p1 == "HIGH", f"Expected HIGH, got {p1}"
    assert p2 == "MEDIUM", f"Expected MEDIUM, got {p2}"
    assert p3 == "LOW", f"Expected LOW, got {p3}"
    print(f"   [OK] Priority detection rules verified (High: {p1}, Medium: {p2}, Low: {p3})")
    
    # 3. RAG Similarity Search
    print("\n[3/6] Testing ChromaDB RAG Vector Store...")
    rag_wifi = query_rag("My WiFi is not connecting", top_k=2)
    assert len(rag_wifi["chunks"]) > 0, "No RAG chunks retrieved for WiFi query!"
    print(f"   [OK] RAG vector store query retrieved {len(rag_wifi['chunks'])} relevant chunks from knowledge base.")
    
    # 4. LangGraph Classifier & Technical Routing (Scenario 1 & 2)
    print("\n[4/6] Testing LangGraph Routing Scenarios...")
    
    # Scenario 1: WiFi
    res1 = run_helpdesk_pipeline("My WiFi is not connecting", user_name="Throna")
    print(f"   [OK] Scenario 1 ('My WiFi is not connecting') -> Classifier: '{res1['classification']}'")
    assert res1['classification'] == "technical"
    
    # Scenario 2: Slow laptop
    res2 = run_helpdesk_pipeline("My laptop is very slow", user_name="Throna")
    print(f"   [OK] Scenario 2 ('My laptop is very slow') -> Classifier: '{res2['classification']}'")
    assert res2['classification'] == "technical"
    
    # 5. Ticket Agent & Human-In-The-Loop Confirmation (Scenario 3 & 4)
    print("\n[5/6] Testing Ticket Creation & Status Lookup...")
    
    # Scenario 3: Create ticket intent
    res3 = run_helpdesk_pipeline("Create a ticket because my laptop keeps restarting", user_name="Throna")
    print(f"   [OK] Scenario 3 ('Create ticket...') -> Classifier: '{res3['classification']}'")
    assert res3['classification'] == "ticket"
    assert res3['pending_ticket'] is not None
    assert res3['pending_ticket']['priority'] == "HIGH"
    print(f"   [OK] Pending HITL Ticket detected with priority: {res3['pending_ticket']['priority']}")
    
    # Execute Ticket Tool Creation (Simulating user clicking YES)
    new_tkt = create_ticket(user_name="Throna", issue="Laptop keeps restarting", priority="HIGH")
    tkt_id = new_tkt["ticket_id"]
    print(f"   [OK] Ticket created in SQLite DB with ID: {tkt_id}")
    
    # Scenario 4: Check Ticket Status
    status_msg = check_ticket_status(tkt_id)
    assert tkt_id in status_msg
    print(f"   [OK] Ticket status lookup verified:\n{status_msg}")
    
    # 6. Long-Term Memory Persistence (Scenario 5)
    print("\n[6/6] Testing Long-Term SQLite Memory...")
    res5_store = run_helpdesk_pipeline("My name is Throna", user_name="User")
    assert res5_store["user_name"] == "Throna"
    
    # Verify retrieval
    stored_name = retrieve_user_fact("user_name")
    assert stored_name == "Throna"
    print(f"   [OK] Memory saved and retrieved across turns: user_name='{stored_name}'")
    
    print("\n=========================================")
    print("  ALL 5 DEMO SCENARIOS PASSED SUCCESSFULLY!")
    print("=========================================")

if __name__ == "__main__":
    run_tests()
