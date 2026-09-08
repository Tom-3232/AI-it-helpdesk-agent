import re
from src.database import save_user_memory, get_user_memory

DEFAULT_USER_ID = "default_user"

def extract_and_store_memories(user_input: str, user_id: str = DEFAULT_USER_ID) -> str:
    """
    Extracts explicit user facts (e.g., 'My name is X', 'I work in IT') and stores them in SQLite.
    Returns a message if a memory was learned.
    """
    input_lower = user_input.lower().strip()
    
    # 1. Pattern: "My name is <Name>"
    name_match = re.search(r"my name is ([a-zA-Z\s]+)", user_input, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r"i am ([a-zA-Z]+)$", user_input, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r"call me ([a-zA-Z\s]+)", user_input, re.IGNORECASE)
        
    if name_match:
        name = name_match.group(1).strip().title()
        # Clean up any trailing punctuation
        name = re.sub(r"[^\w\s]", "", name).strip()
        save_user_memory(user_id, "user_name", name)
        return name
        
    return None

def retrieve_user_fact(memory_key: str, user_id: str = DEFAULT_USER_ID) -> str:
    """Retrieves a specific fact from SQLite long-term memory."""
    memories = get_user_memory(user_id, memory_key)
    if memories:
        return memories[0]["memory_value"]
    return None

def format_all_user_memories(user_id: str = DEFAULT_USER_ID) -> str:
    """Formats all stored long-term memory facts into a summary string for context injection."""
    memories = get_user_memory(user_id)
    if not memories:
        return "No prior user facts stored."
    
    facts = [f"- {m['memory_key'].replace('_', ' ').capitalize()}: {m['memory_value']}" for m in memories]
    return "\n".join(facts)
