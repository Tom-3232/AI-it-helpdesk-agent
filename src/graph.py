from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from src.agents import (
    run_classifier_node,
    run_technical_agent_node,
    run_ticket_agent_node,
    run_general_agent_node
)

class HelpdeskState(TypedDict):
    messages: List[Dict[str, Any]]
    user_name: str
    classification: Optional[str]
    pending_ticket: Optional[Dict[str, Any]]
    retrieved_context: Optional[str]
    response: Optional[str]

def route_by_classification(state: HelpdeskState) -> str:
    """
    Conditional routing edge function.
    Reads state['classification'] and returns target node name.
    """
    category = state.get("classification", "general")
    if category == "technical":
        return "technical_agent"
    elif category == "ticket":
        return "ticket_agent"
    else:
        return "general_agent"

def build_helpdesk_graph():
    """
    Constructs and compiles the explicit LangGraph StateGraph workflow for IT Helpdesk routing.
    
    Workflow Topology:
    
            [START]
               │
        classifier_node
               │
     ┌─────────┼──────────┐
     ▼         ▼          ▼
 technical  ticket    general
   agent     agent     agent
     │         │          │
     └─────────┴──────────┘
               │
             [END]
    """
    workflow = StateGraph(HelpdeskState)
    
    # 1. Add nodes
    workflow.add_node("classifier", run_classifier_node)
    workflow.add_node("technical_agent", run_technical_agent_node)
    workflow.add_node("ticket_agent", run_ticket_agent_node)
    workflow.add_node("general_agent", run_general_agent_node)
    
    # 2. Set Entry Point
    workflow.set_entry_point("classifier")
    
    # 3. Add Conditional Routing Edges
    workflow.add_conditional_edges(
        "classifier",
        route_by_classification,
        {
            "technical_agent": "technical_agent",
            "ticket_agent": "ticket_agent",
            "general_agent": "general_agent"
        }
    )
    
    # 4. Add Edges from Agents to END
    workflow.add_edge("technical_agent", END)
    workflow.add_edge("ticket_agent", END)
    workflow.add_edge("general_agent", END)
    
    return workflow.compile()

# Global compiled graph instance
helpdesk_app = build_helpdesk_graph()

def run_helpdesk_pipeline(query: str, user_name: str = "User", history: List[Dict[str, Any]] = None) -> HelpdeskState:
    """
    Executes the compiled LangGraph pipeline for a single user query.
    """
    if history is None:
        history = []
        
    messages = list(history)
    messages.append({"role": "user", "content": query})
    
    initial_state: HelpdeskState = {
        "messages": messages,
        "user_name": user_name,
        "classification": None,
        "pending_ticket": None,
        "retrieved_context": None,
        "response": None
    }
    
    final_state = helpdesk_app.invoke(initial_state)
    return final_state
