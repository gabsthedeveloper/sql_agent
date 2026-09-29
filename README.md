# SQL Multi-Agent System
## Overview
The purpose of this project was to extract large volumes of data and enable an agent to respond to related questions. Because the system needed to manage massive data without overwhelming the LLM or exhausting tokens, it evolved into a multi-agent system. <br /><br />

This multi-agent system combines a built-in database with two AI agents. It first loads a CSV file and builds the database using DuckDB. The initial agent generates SQL queries from the user’s prompt, and a built-in tool checks each query before passing it forward. The second agent summarizes the results of successful queries to respond to the prompt. The system monitors the chat history to maintain a smooth conversation with the summary agent. <br /><br />

In the most recent update, I have incorporated Logfire for telemetry and evaluations.
