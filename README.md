# ABC Technologies - AI Customer Support Automation System
## Built with LangGraph | IBM Agentic AI Certification Assignment 2

---

## Project Overview

This project implements an AI-powered customer support automation system for ABC Technologies using **LangGraph**. The system intelligently routes customer queries to specialized support agents, retrieves information from a knowledge base using RAG, maintains conversation history with SQLite, and includes a human-in-the-loop approval process for high-risk requests.

---

## Architecture

```
Customer Query
      ↓
[Load Memory Node] ← SQLite (memory.db)
      ↓
[Intent Classification Node]
      ↓ (conditional routing)
┌─────┬──────────┬─────────┬─────────┬────────────┐
│Sales│Technical │ Billing │ Account │   Memory   │
│Agent│  Agent   │  Agent  │  Agent  │Recall Agent│
└─────┴──────────┴─────────┴─────────┴────────────┘
      ↓ (conditional: approval needed?)
[Human Approval Node] ← (for refunds, cancellations, closures)
      ↓
[Supervisor Agent] ← RAG (FAISS Vector Store)
      ↓
Final Response to Customer
```

---

## Features

| Feature | Implementation |
|---------|---------------|
| Intent Classification | GPT-4o-mini classifies into Sales/Technical/Billing/Account/Memory |
| Conditional Routing | LangGraph conditional edges route to correct department |
| 4 Specialized Agents | Sales, Technical, Billing, Account support agents |
| RAG Pipeline | FAISS vector store with 4 knowledge base documents |
| SQLite Memory | Stores and retrieves full conversation history |
| Human-in-the-Loop | Console-based supervisor approval for high-risk requests |
| Supervisor Agent | Reviews and polishes all responses before delivery |

---

## Project Structure

```
customer_support_ai/
├── src/
│   └── main.py              ← Main application (run this!)
├── knowledge_base/
│   ├── company_policy.txt   ← Refund, cancellation, support policies
│   ├── pricing_guide.txt    ← All subscription plans and pricing
│   ├── technical_manual.txt ← Technical troubleshooting guide
│   └── faq.txt              ← Frequently asked questions
├── requirements.txt          ← Python dependencies
├── .env.example             ← Environment variable template
├── .env                     ← Your actual API keys (create this!)
├── memory.db                ← SQLite database (auto-created on first run)
└── README.md                ← This file
```

---

## Setup Instructions

### Step 1: Install Python
- Download Python 3.11+ from https://www.python.org/downloads/
- During installation, check "Add Python to PATH"

### Step 2: Install VS Code
- Download from https://code.visualstudio.com/
- Install the Python extension from VS Code marketplace

### Step 3: Open the Project
```bash
# In VS Code terminal (Terminal > New Terminal):
cd customer_support_ai
```

### Step 4: Create Virtual Environment
```bash
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Set Your OpenAI API Key
```bash
# Copy the example file:
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux

# Open .env in VS Code and replace:
# OPENAI_API_KEY=your-openai-api-key-here
# with your actual key from https://platform.openai.com/api-keys
```

### Step 7: Run the Application
```bash
cd src
python main.py
```

---

## Demo Queries (Assignment Task 10)

| Query | Expected Path |
|-------|--------------|
| "What are the pricing plans available?" | Sales Agent |
| "I forgot my account password" | Account Agent |
| "My application crashes when I upload a file" | Technical Agent |
| "I need a refund for my annual subscription" | Billing → Human Approval |
| "What was my previous support issue?" | Memory Recall Agent |

---

## Human-in-the-Loop

The following request types trigger human supervisor approval:
- Refund requests
- Subscription cancellations  
- Account closure requests
- Compensation requests
- Escalation to management

When triggered, the system will pause and prompt: **"Supervisor Decision - Approve this request? (yes/no)"**

---

## Memory System (SQLite)

The `memory.db` file stores:
- `conversation_history` table: All customer messages and AI responses
- `customer_profiles` table: Customer info and interaction counts

To view the database directly, you can use DB Browser for SQLite (free download from https://sqlitebrowser.org/).

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| LangGraph | Workflow orchestration and state management |
| LangChain | LLM integration and RAG pipeline |
| OpenAI GPT-4o-mini | Language model for all agents |
| FAISS | Vector similarity search for RAG |
| SQLite | Persistent conversation memory |
| Python | Programming language |
