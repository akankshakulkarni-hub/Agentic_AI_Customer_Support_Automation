-- ABC Technologies Customer Support AI
-- SQLite Memory Database Schema
-- This file documents the database structure used by the system

-- Table 1: Stores all customer messages and AI responses
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,    -- Auto-incrementing unique ID
    customer_id TEXT NOT NULL,               -- Customer's unique identifier
    customer_name TEXT,                      -- Customer's display name
    timestamp TEXT NOT NULL,                 -- ISO format: 2025-01-15T10:30:00
    role TEXT NOT NULL,                      -- 'user' (customer) or 'assistant' (AI)
    message TEXT NOT NULL,                   -- The actual message content
    intent TEXT,                             -- Classified intent: Sales/Technical/Billing/Account/Memory
    session_id TEXT                          -- Groups messages from the same session
);

-- Table 2: Stores customer profile information
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id TEXT PRIMARY KEY,            -- Unique identifier for the customer
    customer_name TEXT,                      -- Customer's name
    email TEXT,                              -- Customer's email (optional)
    last_interaction TEXT,                   -- Timestamp of most recent interaction
    total_interactions INTEGER DEFAULT 0     -- Count of all interactions
);

-- Useful queries for reviewing memory data:

-- View all conversations for a specific customer:
-- SELECT * FROM conversation_history WHERE customer_id = 'CUST_001' ORDER BY timestamp;

-- View all customer profiles:
-- SELECT * FROM customer_profiles ORDER BY total_interactions DESC;

-- View recent conversations across all customers:
-- SELECT customer_name, role, message, intent, timestamp
-- FROM conversation_history
-- ORDER BY timestamp DESC LIMIT 20;
