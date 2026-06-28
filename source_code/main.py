"""
ABC Technologies - AI-Powered Customer Support Automation System
Built with LangGraph
"""
import os
import sqlite3
import json
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
import operator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# TASK 2: STATE STRUCTURE
class SupportState(TypedDict):
    customer_id: str           
    customer_name: str             
    query: str                 
    intent: str                
    retrieved_context: str      
    requires_approval: bool    
    approval_status: str      
    approval_reason: str          
    agent_response: str        
    final_response: str        
    conversation_history: str      
    messages: Annotated[List, operator.add]
# RAG PIPELINE SETUP
def setup_rag_pipeline(api_key: str):
    """
    Loads all knowledge base documents and creates a searchable vector store.
    Think of this like creating a smart index for our documents.
    """
    print("\nSetting up RAG Pipeline (loading knowledge base documents)...")
    # Path to knowledge base documents
    kb_path = "knowledge_base"
    documents = []
    # Load each document file
    doc_files = [
        "company_policy.txt",
        "pricing_guide.txt", 
        "technical_manual.txt",
        "faq.txt"
    ]
    for filename in doc_files:
        filepath = os.path.join(kb_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Create a Document object with the content and metadata
            doc = Document(
                page_content=content,
                metadata={"source": filename, "type": "knowledge_base"}
            )
            documents.append(doc)
            print(f"    Loaded: {filename}")
        else:
            print(f"     Not found: {filepath}")
    
    if not documents:
        print("    No documents found! Check knowledge_base/ folder.")
        return None
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       
        chunk_overlap=50,     
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"    Created {len(split_docs)} document chunks for retrieval")
    # Create embeddings (converts text to numbers for similarity search)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    # Create FAISS vector store (stores our document chunks as searchable vectors)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    print("    Vector store created successfully!")
    return vectorstore
def retrieve_relevant_context(vectorstore, query: str, k: int = 4) -> str:
    """
    Given a customer query, find the most relevant information
    from our knowledge base documents.
    
    Args:
        vectorstore: Our searchable document database
        query: The customer's question
        k: How many document chunks to retrieve
    
    Returns:
        A string with the most relevant information
    """
    if vectorstore is None:
        return "Knowledge base not available."
    # Search for relevant documents
    relevant_docs = vectorstore.similarity_search(query, k=k)
    # Combine the retrieved chunks into one context string
    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)
# TASK 7: SQLITE MEMORY SYSTEM
class MemoryManager:
    """
    Manages customer conversation history using SQLite database.
    SQLite is a simple database that stores data in a single file (memory.db).
    """
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._initialize_database()
    def _initialize_database(self):
        """Creates the database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Create table to store conversation history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                customer_name TEXT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                intent TEXT,
                session_id TEXT
            )
        """)  
        # Create table to store customer profiles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_profiles (
                customer_id TEXT PRIMARY KEY,
                customer_name TEXT,
                email TEXT,
                last_interaction TEXT,
                total_interactions INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
        print(f"   SQLite database initialized: {self.db_path}")
    def save_interaction(self, customer_id: str, customer_name: str, 
                          role: str, message: str, intent: str = "", session_id: str = ""):
        """
        Save a single message to the database.
        Args:
            customer_id: Unique identifier for the customer
            customer_name: Customer's name
            role: "user" (customer) or "assistant" (AI)
            message: The actual message text
            intent: What type of query (Sales, Technical, etc.)
            session_id: Groups messages from the same session
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        # Insert the message
        cursor.execute("""
            INSERT INTO conversation_history 
            (customer_id, customer_name, timestamp, role, message, intent, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (customer_id, customer_name, timestamp, role, message, intent, session_id))
        # Update or create customer profile
        cursor.execute("""
            INSERT INTO customer_profiles (customer_id, customer_name, last_interaction, total_interactions)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(customer_id) DO UPDATE SET
                customer_name = excluded.customer_name,
                last_interaction = excluded.last_interaction,
                total_interactions = total_interactions + 1
        """, (customer_id, customer_name, timestamp))
        conn.commit()
        conn.close()
    def get_conversation_history(self, customer_id: str, limit: int = 10) -> str:
        """
        Retrieve the last N interactions for a customer.
        
        Args:
            customer_id: The customer to look up
            limit: How many past messages to retrieve
        
        Returns:
            A formatted string of conversation history
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, role, message, intent
            FROM conversation_history
            WHERE customer_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (customer_id, limit))
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return "No previous conversation history found."
        # Format history (reversed to show oldest first)
        history_lines = []
        for timestamp, role, message, intent in reversed(rows):
            time_str = timestamp[:19].replace("T", " ")  # Format timestamp
            role_label = "Customer" if role == "user" else "Support AI"
            intent_str = f" [{intent}]" if intent else ""
            history_lines.append(f"[{time_str}]{intent_str} {role_label}: {message}")
        return "\n".join(history_lines)
    def get_customer_name(self, customer_id: str) -> Optional[str]:
        """Look up a customer's name from their ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT customer_name FROM customer_profiles WHERE customer_id = ?", 
                      (customer_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
# INTENT CLASSIFICATION NODE
def classify_intent(state: SupportState, llm: ChatGoogleGenerativeAI) -> SupportState:
    """
    Analyzes the customer's query and classifies it into one of:
    - Sales: pricing, plans, product info
    - Technical: app errors, installation, configuration
    - Billing: invoices, payments, refunds
    - Account: password, profile, activation
    - Memory: asking about previous interactions
    """
    print(f"\n [Intent Classification] Analyzing query: '{state['query'][:60]}...'")
    system_prompt = """You are an intent classifier for ABC Technologies customer support.
Classify the customer query into EXACTLY ONE of these categories:
- Sales: Questions about pricing, subscription plans, product features, demos
- Technical: App errors, crashes, installation problems, login issues, configuration
- Billing: Invoice requests, payment issues, refund requests, subscription management
- Account: Password reset, profile updates, account activation or deactivation
- Memory: Customer asking about their previous support interactions or history
Respond with ONLY the category name, nothing else.
Examples:
- "What are your pricing plans?" → Sales
- "My app keeps crashing" → Technical  
- "I want a refund" → Billing
- "How do I reset my password?" → Account
- "What was my last issue?" → Memory"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Customer Query: {state['query']}")
    ]
    response = llm.invoke(messages)
    intent = response.content.strip()
    # Validate the intent is one of our expected values
    valid_intents = ["Sales", "Technical", "Billing", "Account", "Memory"]
    if intent not in valid_intents:
        # Default to Technical if unclear
        intent = "Technical"
    print(f"    Intent classified as: {intent}")
    return {**state, "intent": intent}
#  CONDITIONAL ROUTING
def route_to_department(state: SupportState) -> str:
    """
    Based on the classified intent, returns the name of the
    next node to execute in our workflow graph.
    
    This function is used as the "conditional edge" in LangGraph.
    """
    intent = state.get("intent", "Technical")
    routing_map = {
        "Sales": "sales_agent",
        "Technical": "technical_agent",
        "Billing": "billing_agent",
        "Account": "account_agent",
        "Memory": "memory_recall_agent"
    }
    destination = routing_map.get(intent, "technical_agent")
    print(f"\n [Router] Routing to: {destination}")
    return destination
# TASK 5: SPECIALIZED DEPARTMENT AGENTS
def sales_agent(state: SupportState, llm: ChatGoogleGenerativeAI, vectorstore) -> SupportState:
    """Sales Support Agent - handles pricing, plans, product info"""
    print(f"\n [Sales Agent] Processing query...")
    # Retrieve relevant context from knowledge base (RAG)
    context = retrieve_relevant_context(
        vectorstore, 
        state["query"] + " pricing plans subscription",
        k=4
    )
    system_prompt = f"""You are a helpful Sales Support Agent for ABC Technologies.
Your job is to answer questions about pricing, subscription plans, and product features.
KNOWLEDGE BASE CONTEXT (use this to answer accurately):
{context}
CUSTOMER HISTORY:
{state.get('conversation_history', 'No previous history.')}
Instructions:
- Be friendly, professional, and helpful
- Use the context above to give accurate information
- If asked about pricing, always mention all available plans
- Encourage the customer to try the free trial if relevant
- End with an offer to help further"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)
    print(f"   Sales agent response generated")
    return {
        **state,
        "retrieved_context": context,
        "agent_response": response.content,
        "requires_approval": False,
        "approval_status": "not_required"
    }
def technical_agent(state: SupportState, llm: ChatGoogleGenerativeAI, vectorstore) -> SupportState:
    """Technical Support Agent - handles errors, crashes, configuration"""
    print(f"\n [Technical Agent] Processing query...")
    # Retrieve relevant context
    context = retrieve_relevant_context(
        vectorstore,
        state["query"] + " technical error solution fix",
        k=4
    )
    system_prompt = f"""You are a Technical Support Agent for ABC Technologies.
Your job is to help customers resolve technical issues with the software.
TECHNICAL KNOWLEDGE BASE (use this to give accurate solutions):
{context}
CUSTOMER HISTORY:
{state.get('conversation_history', 'No previous history.')}
Instructions:
- Be patient and clear in your explanations
- Provide step-by-step solutions
- Use the knowledge base to give accurate technical guidance
- If the issue seems complex, tell the customer you'll escalate to a senior tech
- Always ask if the solution resolved their issue
- For crashes: ask for browser, OS, and steps to reproduce"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)
    print(f"    Technical agent response generated")
    return {
        **state,
        "retrieved_context": context,
        "agent_response": response.content,
        "requires_approval": False,
        "approval_status": "not_required"
    }
def billing_agent(state: SupportState, llm: ChatGoogleGenerativeAI, vectorstore) -> SupportState:
    """
    Billing Support Agent - handles invoices, payments, refunds.
    IMPORTANT: Refund and cancellation requests require human approval.
    """
    print(f"\n [Billing Agent] Processing query...")
    # Retrieve relevant context
    context = retrieve_relevant_context(
        vectorstore,
        state["query"] + " billing payment refund invoice",
        k=4
    )
    # Check if this query requires human approval
    # These trigger words indicate high-risk requests
    high_risk_keywords = [
        "refund", "cancel", "cancellation", "subscription cancel",
        "money back", "reimburse", "annual subscription refund",
        "compensation", "credit", "account closure", "close account"
    ]
    query_lower = state["query"].lower()
    requires_approval = any(keyword in query_lower for keyword in high_risk_keywords)
    if requires_approval:
        # Determine the specific reason for approval
        if "refund" in query_lower or "money back" in query_lower or "reimburse" in query_lower:
            approval_reason = "Refund Request"
        elif "cancel" in query_lower:
            approval_reason = "Subscription Cancellation"
        elif "close account" in query_lower or "account closure" in query_lower:
            approval_reason = "Account Closure Request"
        elif "compensation" in query_lower or "credit" in query_lower:
            approval_reason = "Compensation Request"
        else:
            approval_reason = "High-Risk Billing Request"
        print(f"     HIGH-RISK REQUEST DETECTED: {approval_reason} - Requires human approval!")
        
        system_prompt = f"""You are a Billing Support Agent for ABC Technologies.
This customer has made a request that requires human supervisor approval.
BILLING POLICY (use this for accurate information):
{context}
CUSTOMER HISTORY:
{state.get('conversation_history', 'No previous history.')}
Instructions:
- Acknowledge the customer's request with empathy
- Explain that their request ({approval_reason}) requires supervisor review
- Tell them the process: their request has been flagged for human supervisor approval
- Mention the expected timeline (1-2 business days)
- Assure them their request is being taken seriously
- Do NOT promise approval or rejection at this stage
- Be professional and reassuring"""
    else:
        approval_reason = ""
        system_prompt = f"""You are a Billing Support Agent for ABC Technologies.
Help the customer with their billing inquiry.
BILLING KNOWLEDGE BASE:
{context}
CUSTOMER HISTORY:
{state.get('conversation_history', 'No previous history.')}
Instructions:
- Help with invoice requests, payment questions, plan information
- Use the knowledge base for accurate billing information
- Be clear about payment methods, billing cycles, and policies
- Direct to the billing portal for self-service options"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)
    print(f"    Billing agent response generated")
    return {
        **state,
        "retrieved_context": context,
        "agent_response": response.content,
        "requires_approval": requires_approval,
        "approval_status": "pending" if requires_approval else "not_required",
        "approval_reason": approval_reason
    }
def account_agent(state: SupportState, llm: ChatGoogleGenerativeAI, vectorstore) -> SupportState:
    """Account Support Agent - handles password reset, profile, activation"""
    print(f"\n [Account Agent] Processing query...")
    # Retrieve relevant context
    context = retrieve_relevant_context(
        vectorstore,
        state["query"] + " account password profile activation",
        k=4
    )
    # Check for account closure (requires approval)
    query_lower = state["query"].lower()
    requires_approval = "close account" in query_lower or "delete account" in query_lower or "account closure" in query_lower
    if requires_approval:
        approval_reason = "Account Closure Request"
        print(f"     ACCOUNT CLOSURE REQUEST - Requires human approval!")
    else:
        approval_reason = ""
    system_prompt = f"""You are an Account Support Agent for ABC Technologies.
Your job is to help customers with account-related issues.
ACCOUNT MANAGEMENT KNOWLEDGE BASE:
{context}
CUSTOMER HISTORY:
{state.get('conversation_history', 'No previous history.')}
Instructions:
- Help with password resets, profile updates, account activation/deactivation
- Provide clear, step-by-step instructions
- For password reset: direct to the "Forgot Password" link on the login page
- For profile updates: guide to Settings > Profile
- Always verify the customer's identity before making sensitive changes
- Be security-conscious - never ask for passwords directly"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)
    print(f"    Account agent response generated")
    return {
        **state,
        "retrieved_context": context,
        "agent_response": response.content,
        "requires_approval": requires_approval,
        "approval_status": "pending" if requires_approval else "not_required",
        "approval_reason": approval_reason
    }
def memory_recall_agent(state: SupportState, llm: ChatGoogleGenerativeAI, memory_manager: MemoryManager) -> SupportState:
    """
    Memory Recall Agent - retrieves and answers questions about 
    the customer's previous support interactions.
    This agent does NOT route to any department - it answers directly.
    """
    print(f"\n [Memory Recall Agent] Retrieving conversation history...")
    history = state.get("conversation_history", "No previous history.")
    system_prompt = f"""You are a helpful support agent for ABC Technologies.
The customer is asking about their previous support interactions.
THEIR CONVERSATION HISTORY:
{history}
Instructions:
- Answer the customer's question using their history above
- Be helpful and specific about what issues they previously raised
- If they had multiple issues, list them clearly
- If there is no history, politely let them know
- Offer to help with any new issues they might have"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)
    print(f"    Memory recall response generated")
    return {
        **state,
        "retrieved_context": history,
        "agent_response": response.content,
        "requires_approval": False,
        "approval_status": "not_required"
    }
# TASK 8: HUMAN-IN-THE-LOOP APPROVAL
def human_approval_node(state: SupportState) -> SupportState:
    """
    This node pauses the workflow and asks a human supervisor
    to approve or reject high-risk requests.
    
    In production, this would integrate with email/Slack/ticketing system.
    For this demo, it simulates the approval via console input.
    """
    if not state.get("requires_approval", False):
        # No approval needed, skip this node
        return state
    print("\n" + "="*60)
    print(" HUMAN SUPERVISOR APPROVAL REQUIRED")
    print("="*60)
    print(f"Customer ID    : {state['customer_id']}")
    print(f"Customer Name  : {state['customer_name']}")
    print(f"Request Type   : {state['approval_reason']}")
    print(f"Customer Query : {state['query']}")
    print(f"\nAI Draft Response Preview:")
    print("-"*40)
    print(state['agent_response'][:300] + "..." if len(state['agent_response']) > 300 else state['agent_response'])
    print("="*60)
    # Ask supervisor for decision
    while True:
        decision = input("\n Supervisor Decision - Approve this request? (yes/no): ").strip().lower()
        if decision in ["yes", "y"]:
            print("    Request APPROVED by supervisor")
            return {
                **state,
                "approval_status": "approved"
            }
        elif decision in ["no", "n"]:
            rejection_reason = input("   Please enter rejection reason: ").strip()
            print("    Request REJECTED by supervisor")
            return {
                **state,
                "approval_status": "rejected",
                "approval_reason": f"Rejected: {rejection_reason}"
            }
        else:
            print("     Please enter 'yes' or 'no'")
def check_approval_needed(state: SupportState) -> str:
    """
    Conditional edge: checks if human approval is needed.
    Returns the name of the next node to go to.
    """
    if state.get("requires_approval", False):
        return "human_approval"
    else:
        return "supervisor_agent"
# TASK 9: SUPERVISOR AGENT
def supervisor_agent(state: SupportState, llm: ChatGoogleGenerativeAI) -> SupportState:
    """
    The Supervisor Agent reviews the department agent's response
    and produces a final, polished response to send to the customer.
    
    It also handles the case where a request was rejected by the human supervisor.
    """
    print(f"\n [Supervisor Agent] Reviewing and finalizing response...")
    # Handle rejected requests specially
    if state.get("approval_status") == "rejected":
        system_prompt = f"""You are a senior customer support supervisor at ABC Technologies.
A customer's {state['approval_reason']} has been reviewed and cannot be processed at this time.
Original Query: {state['query']}
AI Agent Response: {state['agent_response']}
Write a professional, empathetic response to the customer explaining:
1. Their request has been reviewed
2. Unfortunately it cannot be approved at this time
3. Offer alternative options or next steps they can take
4. Provide contact information for further escalation if needed

Be empathetic, professional, and solution-oriented."""
    elif state.get("approval_status") == "approved":
        system_prompt = f"""You are a senior customer support supervisor at ABC Technologies.
A customer's request has been reviewed and APPROVED by a human supervisor.
Customer Query: {state['query']}
Request Type: {state['approval_reason']}
AI Agent Draft: {state['agent_response']}
Write the final response confirming:
1. Their request has been approved
2. What will happen next and the timeline
3. Any reference number or confirmation details
4. Next steps for the customer
Be professional, clear, and reassuring."""
    else:
        # Standard case - no approval needed, just polish the response
        system_prompt = f"""You are a senior customer support supervisor at ABC Technologies.
Review the AI agent's response and create the final, polished version to send to the customer.
Original Customer Query: {state['query']}
Department: {state['intent']}
AI Agent Draft Response:
{state['agent_response']}

Instructions for improvement:
1. Ensure the response is professional and empathetic
2. Make sure all technical information is clearly explained
3. Add a warm greeting using the customer's name if available
4. Ensure there's a clear call-to-action or next step
5. Add a professional closing
6. Keep it concise but complete (not too long)
7. Fix any grammar or tone issues

Customer Name: {state.get('customer_name', 'Valued Customer')}"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please finalize the response for this customer.")
    ]
    response = llm.invoke(messages)
    print(f"   Final response prepared by supervisor")
    return {
        **state,
        "final_response": response.content
    }
# MEMORY LOADING NODE
def load_customer_memory(state: SupportState, memory_manager: MemoryManager) -> SupportState:
    """
    Loads the customer's conversation history from SQLite
    before the query is processed by any agent.
    """
    customer_id = state.get("customer_id", "unknown")
    history = memory_manager.get_conversation_history(customer_id, limit=10)
    # Also check if we can get the customer's name from history
    stored_name = memory_manager.get_customer_name(customer_id)
    customer_name = state.get("customer_name", stored_name or "Valued Customer")
    return {
        **state,
        "conversation_history": history,
        "customer_name": customer_name
    }
# BUILD THE LANGGRAPH WORKFLOW
def build_workflow(llm: ChatGoogleGenerativeAI, vectorstore, memory_manager: MemoryManager):
    """
    Assembles the complete LangGraph workflow by:
    1. Creating a StateGraph
    2. Adding all nodes (functions)
    3. Adding edges (connections between nodes)
    4. Compiling the graph
    """
    print("\n  Building LangGraph workflow...")
    # Create the state graph
    workflow = StateGraph(SupportState)
    # Node 1: Load customer memory (always runs first)
    workflow.add_node("load_memory", 
                      lambda state: load_customer_memory(state, memory_manager))
    # Node 2: Classify intent
    workflow.add_node("classify_intent", 
                      lambda state: classify_intent(state, llm))
    # Node 3: Department Agents
    workflow.add_node("sales_agent", 
                      lambda state: sales_agent(state, llm, vectorstore))
    workflow.add_node("technical_agent", 
                      lambda state: technical_agent(state, llm, vectorstore))
    workflow.add_node("billing_agent", 
                      lambda state: billing_agent(state, llm, vectorstore))
    workflow.add_node("account_agent", 
                      lambda state: account_agent(state, llm, vectorstore))
    workflow.add_node("memory_recall_agent", 
                      lambda state: memory_recall_agent(state, llm, memory_manager))
    # Node 4: Human approval (for high-risk requests)
    workflow.add_node("human_approval", human_approval_node)
    # Node 5: Supervisor (final response generation)
    workflow.add_node("supervisor_agent", 
                      lambda state: supervisor_agent(state, llm))
    # ADD EDGES (connections between nodes)
    workflow.set_entry_point("load_memory")
    # Load Memory → Classify Intent
    workflow.add_edge("load_memory", "classify_intent")
    # Classify Intent → Route to Department (CONDITIONAL)
    workflow.add_conditional_edges(
        "classify_intent",          # From this node
        route_to_department,        # Use this function to decide
        {                           # Mapping: function return value → next node
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_recall_agent": "memory_recall_agent"
        }
    )
    # Department Agents → Check if Approval Needed (CONDITIONAL)
    for agent in ["sales_agent", "technical_agent", "billing_agent", "account_agent", "memory_recall_agent"]:
        workflow.add_conditional_edges(
            agent,
            check_approval_needed,
            {
                "human_approval": "human_approval",
                "supervisor_agent": "supervisor_agent"
            }
        )
    # Human Approval → Supervisor
    workflow.add_edge("human_approval", "supervisor_agent")
    # Supervisor → END
    workflow.add_edge("supervisor_agent", END)
    # Compile the graph
    app = workflow.compile()
    print("    Workflow compiled successfully!")
    return app
# MAIN DEMO FUNCTION
def run_demo(app, memory_manager: MemoryManager):
    """
    Demonstrates the system using the 5 sample queries from the assignment.
    Each query goes through the complete workflow.
    """
    print("\n" + "="*70)
    print(" ABC TECHNOLOGIES - AI CUSTOMER SUPPORT SYSTEM")
    print("   Assignment Demo - 5 Sample Queries")
    print("="*70)
    # Define the 5 sample queries
    demo_queries = [
        {
            "query_num": 1,
            "customer_id": "CUST_001",
            "customer_name": "Alice Johnson",
            "query": "What are the pricing plans available for your software?",
            "expected_path": "Sales"
        },
        {
            "query_num": 2,
            "customer_id": "CUST_002",
            "customer_name": "Bob Smith",
            "query": "I forgot my account password and cannot log in. How do I reset it?",
            "expected_path": "Account"
        },
        {
            "query_num": 3,
            "customer_id": "CUST_003",
            "customer_name": "Carol White",
            "query": "My application crashes whenever I try to upload a file. This is very frustrating!",
            "expected_path": "Technical Support"
        },
        {
            "query_num": 4,
            "customer_id": "CUST_004",
            "customer_name": "David Brown",
            "query": "I need a refund for my annual subscription. I purchased it last week but want to cancel.",
            "expected_path": "Billing → Human Approval Required"
        },
        {
            "query_num": 5,
            "customer_id": "CUST_001",  # Same as Query 1 to test memory recall
            "customer_name": "Alice Johnson",
            "query": "What was my previous support issue?",
            "expected_path": "Memory Recall"
        }
    ]
    
    # Process each query
    for demo in demo_queries:
        print(f"\n{'='*70}")
        print(f" QUERY {demo['query_num']} OF 5")
        print(f"   Customer    : {demo['customer_name']} (ID: {demo['customer_id']})")
        print(f"   Query       : {demo['query']}")
        print(f"   Expected    : {demo['expected_path']}")
        print(f"{'='*70}")
        # Save customer's query to memory BEFORE processing
        memory_manager.save_interaction(
            customer_id=demo['customer_id'],
            customer_name=demo['customer_name'],
            role="user",
            message=demo['query'],
            session_id=f"DEMO_{demo['query_num']}"
        )
        # Create the initial state
        initial_state: SupportState = {
            "customer_id": demo['customer_id'],
            "customer_name": demo['customer_name'],
            "query": demo['query'],
            "intent": "",
            "retrieved_context": "",
            "requires_approval": False,
            "approval_status": "not_required",
            "approval_reason": "",
            "agent_response": "",
            "final_response": "",
            "conversation_history": "",
            "messages": []
        }
        # Run the workflow
        result = app.invoke(initial_state)
        # Save AI response to memory AFTER processing
        memory_manager.save_interaction(
            customer_id=demo['customer_id'],
            customer_name=demo['customer_name'],
            role="assistant",
            message=result.get("final_response", "")[:500],  # Save first 500 chars
            intent=result.get("intent", ""),
            session_id=f"DEMO_{demo['query_num']}"
        )
        # Display the final result
        print(f"\n RESULT SUMMARY:")
        print(f"   Intent Detected  : {result.get('intent', 'N/A')}")
        print(f"   Approval Required: {result.get('requires_approval', False)}")
        print(f"   Approval Status  : {result.get('approval_status', 'N/A')}")
        
        print(f"\n FINAL RESPONSE TO CUSTOMER:")
        print("-"*60)
        print(result.get('final_response', 'No response generated.'))
        print("-"*60)
        
        input("\n  Press ENTER to continue to next query...")
# INTERACTIVE MODE
def run_interactive_mode(app, memory_manager: MemoryManager):
    """
    Interactive mode: allows you to type your own queries and
    see the system respond in real-time.
    """
    print("\n" + "="*60)
    print(" INTERACTIVE MODE")
    print("   Type your customer queries to test the system.")
    print("   Type 'quit' to exit.")
    print("="*60)
    customer_id = input("\nEnter your Customer ID (e.g., CUST_001): ").strip() or "CUST_TEST"
    customer_name = input("Enter your name: ").strip() or "Test Customer"
    print(f"\nWelcome, {customer_name}! How can I help you today?")
    while True:
        print("\n" + "-"*40)
        query = input("You: ").strip()
        if query.lower() in ["quit", "exit", "bye"]:
            print("Thank you for contacting ABC Technologies. Goodbye!")
            break
        if not query:
            continue
        # Save query to memory
        memory_manager.save_interaction(
            customer_id=customer_id,
            customer_name=customer_name,
            role="user",
            message=query
        )
        # Create initial state
        initial_state: SupportState = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "query": query,
            "intent": "",
            "retrieved_context": "",
            "requires_approval": False,
            "approval_status": "not_required",
            "approval_reason": "",
            "agent_response": "",
            "final_response": "",
            "conversation_history": "",
            "messages": []
        }
        # Run workflow
        result = app.invoke(initial_state)
        # Save response to memory
        memory_manager.save_interaction(
            customer_id=customer_id,
            customer_name=customer_name,
            role="assistant",
            message=result.get("final_response", "")[:500],
            intent=result.get("intent", "")
        )
        print(f"\nSupport AI [{result.get('intent', '')}]: {result.get('final_response', 'Sorry, I could not process your request.')}")
# ENTRY POINT
def main():
    """
    Main entry point. Sets up the system and runs the demo.
    """
    print("\n" + "="*60)
    print("  ABC Technologies Customer Support AI System")
    print("  Powered by LangGraph + Gemini + FAISS + SQLite")
    print("="*60)
    # Check for API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n ERROR: GEMINI_API_KEY environment variable not set!")
        print("   Please set it in your .env file")
        return
    print("\n Gemini API key found.")
    # Initialize LLM
    print("\n Initializing LLM (Gemini 2.5 Flash)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,        # Lower = more consistent, less creative
        google_api_key=api_key
    )
    # Setup RAG pipeline
    vectorstore = setup_rag_pipeline(api_key)
    # Initialize memory manager
    print("\n Initializing SQLite memory manager...")
    memory_manager = MemoryManager(db_path="memory.db")
    # Build the LangGraph workflow
    app = build_workflow(llm, vectorstore, memory_manager)
    # Ask user what mode to run
    print("\n" + "="*60)
    print("Choose mode:")
    print("  1 - Run Demo (5 sample queries from assignment)")
    print("  2 - Interactive Mode (type your own queries)")
    print("="*60)
    choice = input("Enter choice (1 or 2): ").strip()
    if choice == "1":
        run_demo(app, memory_manager)
    elif choice == "2":
        run_interactive_mode(app, memory_manager)
    else:
        print("Invalid choice. Running demo mode...")
        run_demo(app, memory_manager)
    print("\n Session complete. Goodbye!")
if __name__ == "__main__":
    main()