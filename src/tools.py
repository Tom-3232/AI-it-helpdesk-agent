import datetime
import random
from langchain_core.tools import tool
from src.database import insert_ticket, get_ticket

def detect_priority(issue_text: str) -> str:
    """
    Analyzes issue description and determines priority badge (HIGH, MEDIUM, LOW).
    Rule-based priority matching:
    - HIGH: laptop restarting, system down, security issue, data loss, critical failure
    - MEDIUM: wifi, general software, vpn, email, network
    - LOW: printer, general requests, minor issues
    """
    text = issue_text.lower()
    
    high_keywords = [
        "restarting", "restart", "rebooting", "system down", "completely down",
        "security issue", "data loss", "critical failure", "crash loop",
        "blue screen", "bsod", "hacked", "ransomware", "cannot boot"
    ]
    medium_keywords = [
        "wifi", "wi-fi", "internet", "software", "vpn", "email",
        "outlook", "slow", "network", "connection", "login", "password"
    ]
    
    for kw in high_keywords:
        if kw in text:
            return "HIGH"
            
    for kw in medium_keywords:
        if kw in text:
            return "MEDIUM"
            
    return "LOW"

@tool
def create_ticket_tool(user_name: str, issue: str, priority: str = "MEDIUM") -> dict:
    """
    Creates an IT support ticket in SQLite database with status OPEN and a unique Ticket ID.
    Args:
        user_name: Name of the user submitting the ticket.
        issue: Description of the technical issue or request.
        priority: Ticket priority ('HIGH', 'MEDIUM', or 'LOW').
    """
    num = random.randint(10000, 99999)
    ticket_id = f"TKT-2026-{num}"
    ticket = insert_ticket(user_name=user_name, issue=issue, priority=priority, ticket_id=ticket_id)
    return ticket

@tool
def check_ticket_status_tool(ticket_id: str) -> str:
    """
    Checks the status of an existing IT support ticket by ticket ID.
    Args:
        ticket_id: Unique ticket ID string (e.g. 'TKT-2026-12345').
    """
    ticket_id_clean = ticket_id.strip().upper()
    ticket = get_ticket(ticket_id_clean)
    if ticket:
        return (
            f"🎫 Ticket Details Found:\n"
            f"• Ticket ID: {ticket['ticket_id']}\n"
            f"• User: {ticket['user_name']}\n"
            f"• Issue: {ticket['issue']}\n"
            f"• Priority: {ticket['priority']}\n"
            f"• Status: {ticket['status']}\n"
            f"• Created At: {ticket['created_at']}"
        )
    return f"❌ Ticket ID '{ticket_id_clean}' was not found in the database. Please check the ticket number."

@tool
def get_current_time_tool() -> str:
    """
    Returns the current date and time.
    """
    now = datetime.datetime.now()
    return f"Current Date and Time: {now.strftime('%A, %B %d, %Y %I:%M %p')}"

# Export plain python functions alongside tool objects
def create_ticket(user_name: str, issue: str, priority: str) -> dict:
    return create_ticket_tool.invoke({"user_name": user_name, "issue": issue, "priority": priority})

def check_ticket_status(ticket_id: str) -> str:
    return check_ticket_status_tool.invoke({"ticket_id": ticket_id})

def get_current_time() -> str:
    return get_current_time_tool.invoke({})
