# 🖥️ AI IT Helpdesk Agent

An enterprise-grade, 100% local, zero-cost **Agentic AI IT Helpdesk Agent** built with **LangGraph**, **LangChain**, **ChromaDB**, **SQLite**, and **Streamlit**.

This project provides an end-to-end automated IT support solution featuring explicit workflow routing, Retrieval-Augmented Generation (RAG) over corporate knowledge bases, tool calling for ticket management, long-term memory persistence across app restarts, and Human-in-the-Loop (HITL) confirmation for support ticket submissions.

---

## 📌 Problem Statement

IT helpdesks face overwhelming volumes of repetitive user queries ranging from basic Wi-Fi and printer troubleshooting to account lockouts and hardware failures. Manual ticketing systems often result in delayed response times, poor prioritization, and repetitive data entry. 

## 💡 Solution

The **AI IT Helpdesk Agent** acts as an autonomous virtual IT specialist. It uses a **LangGraph StateGraph** classifier to route requests dynamically:
- **Technical Queries** are grounded with RAG over indexed knowledge base docs.
- **Ticket Requests** undergo rule-based priority analysis and trigger Human-in-the-Loop confirmation before recording tickets in SQLite.
- **General Queries & Facts** update and retrieve long-term user memory profiles.

---

## 🏗️ System Architecture & Workflow

### LangGraph Workflow Topology

```
                  ┌──────────────────────┐
                  │    User Query        │
                  └──────────┬───────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │  classifier_node  │
                   └─────────┬─────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
    ┌────────────────┐ ┌───────────┐ ┌──────────────┐
    │technical_agent │ │ticket_agent│ │general_agent │
    └────────┬───────┘ └─────┬─────┘ └──────┬───────┘
             │               │              │
             └───────────────┼──────────────┘
                             │
                             ▼
                          [ END ]
```

### RAG Pipeline Flow
1. **Document Loading**: Text loaders scan `knowledge_base/` (`wifi.txt`, `slow_laptop.txt`, `vpn.txt`, etc.).
2. **Chunking**: `RecursiveCharacterTextSplitter` creates 500-character overlapping chunks.
3. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` produces dense vector representations.
4. **Vector Store**: `ChromaDB` stores embeddings locally in `data/chroma_db/`.
5. **Contextual Generation**: `ChatOllama` (`llama3.2`) generates grounded answers referencing retrieved context.

---

## 🚀 Features

- **100% Free & Local Stack**: Zero API key dependencies. Runs via Ollama, ChromaDB, HuggingFace embeddings, SQLite, and Streamlit.
- **Live Ollama Health Monitoring**: Real-time status banner indicating if Ollama and `llama3.2` are online/offline.
- **Human-in-the-Loop (HITL) Safeguards**: Confirms ticket creation with `YES - Create Ticket` or `NO - Cancel` before performing SQLite writes.
- **Color-Coded Priority Detection**: Automatically classifies issues into `HIGH` (Red), `MEDIUM` (Orange), and `LOW` (Green) priority badges.
- **Long-Term SQLite Memory**: Persists user facts (such as names and past context) in `data/helpdesk.db` across sessions.
- **Tool Calling**: Native callable tools for ticket insertion (`create_ticket`), ticket status lookup (`check_ticket_status`), and time retrieval (`get_current_time`).

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Glassmorphic dark dashboard)
- **Agent Orchestration**: LangGraph (`StateGraph`), LangChain
- **LLM Engine**: Ollama (`llama3.2`)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Database**: ChromaDB (Persisted locally)
- **Relational DB & Memory**: SQLite (`data/helpdesk.db`)
- **Language**: Python 3.10+

---

## 📂 Project Structure

```
ai-it-helpdesk-agent/
├── app.py                  # Streamlit dashboard UI & HITL interface
├── requirements.txt        # Python dependency specifications
├── README.md               # Documentation & architecture guide
├── test_helpdesk.py        # Verification test suite covering all 5 demo scenarios
├── .gitignore              # Data and cache ignores
├── src/
│   ├── __init__.py         # Package init
│   ├── config.py           # Ollama health check & host settings
│   ├── database.py         # SQLite connection & CRUD operations
│   ├── memory.py           # Short-term and long-term user memory
│   ├── tools.py            # Ticket tools & priority detection
│   ├── rag.py              # ChromaDB vector index & retrieval
│   ├── agents.py           # LangGraph node execution functions
│   └── graph.py            # LangGraph StateGraph & conditional routing
├── knowledge_base/         # IT troubleshooting guides
│   ├── wifi.txt
│   ├── internet.txt
│   ├── printer.txt
│   ├── software.txt
│   ├── password.txt
│   ├── vpn.txt
│   ├── email.txt
│   └── slow_laptop.txt
└── data/
    └── .gitkeep            # Storage directory for helpdesk.db and ChromaDB
```

---

## ⚡ Installation & Setup

### 1. Clone & Install Dependencies
```bash
cd ai-it-helpdesk-agent
pip install -r requirements.txt
```

### 2. Ollama Setup (Local LLM Engine)
Install [Ollama](https://ollama.com) and pull the model:
```bash
ollama pull llama3.2
```

---

## 🖥️ Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

Run the end-to-end test suite:
```bash
python test_helpdesk.py
```

---

## 🎬 Verified Demo Scenarios

| Scenario | Input Query | Agent Route | Outcome |
| :--- | :--- | :--- | :--- |
| **1. Wi-Fi Issue** | `"My WiFi is not connecting"` | `technical_agent` | RAG retrieves `wifi.txt` context; presents step-by-step fix. |
| **2. Slow Laptop** | `"My laptop is very slow"` | `technical_agent` | RAG retrieves `slow_laptop.txt` context; presents CPU/RAM cleanup guide. |
| **3. High Priority Ticket** | `"Create a ticket because my laptop keeps restarting"` | `ticket_agent` | Detects `HIGH` priority -> Triggers HITL confirmation -> On `YES` inserts into SQLite. |
| **4. Ticket Lookup** | `"Check ticket status TKT-2026-XXXXX"` | `ticket_agent` | Invokes `check_ticket_status` tool -> Returns status, date, priority from SQLite. |
| **5. Long-Term Memory** | `"My name is Throna"` -> `"What is my name?"` | `general_agent` | Stores fact in `user_memory` SQLite table; retrieves correctly across restarts. |

---

## 🚀 Future Improvements

- Integrate active Directory / LDAP single sign-on (SSO).
- Add automated email notification dispatch upon ticket creation.
- Support multi-modal screenshot diagnostics via vision LLMs.
