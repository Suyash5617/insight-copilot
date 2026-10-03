# Insight Copilot — Design Decisions & Assignment Write-Up

## Executive Summary
**Insight Copilot** is a reasoning-first conversational data analyst built for the Generative AI / LLM / Agentic AI Internship Assignment. It transforms natural language queries on the real **Global SuperStore Sales Dataset** (`data/SuperStore_Sales_Dataset.csv`) into structured, executive-ready business insights using **LangGraph StateGraph**.

---

## 1. Problem & Approach

**The Problem**: standard LLM applications often attempt to answer data questions in a single opaque call. This results in hallucinated numbers, incorrect aggregations, and uninspectable reasoning.

**Our Approach**: We built a deterministic multi-step agent using **LangGraph StateGraph** that enforces a strict separation of concerns:
1. **Planning**: Articulate a short user-facing plan before executing code.
2. **Conditional Routing**: Route dataset queries to tools while answering non-data prompts directly.
3. **Empirical Tool Execution**: Compute numbers using Pandas/Plotly engines on the real CSV file.
4. **Synthesis**: Convert raw tables into a structured analyst report (**Answer**, **Supporting Numbers**, **Why This Matters**).

---

## 2. Why LangGraph Was Used

LangGraph provided the core framework for constructing a state machine with explicit control flow:
- **Typed State Management**: `AgentState` carries structured context across nodes.
- **Conditional Routing**: Enables genuine routing decisions based on state inspection (`requires_tool`), preventing arbitrary execution loops.
- **Inspectability**: Each node transition is observable, making agent decisions fully auditable.

---

## 3. Tool Selection & User-Facing Rationale

- **4 Specialized Tools**: `query_dataset`, `calculate_statistics`, `generate_visualization`, and `get_schema_and_summary`.
- **User-Facing Plan**: Instead of exposing private LLM chain-of-thought, the agent displays a concise plan (WHAT it intends to do) in a collapsible **📋 Agent Plan** section in the UI.

---

## 4. Real Dataset Integrity

We used the real **SuperStore Sales Dataset** (`data/SuperStore_Sales_Dataset.csv`), containing 5,901 order records spanning 2019-2020. No fake or synthetic dataset was generated, and the original CSV remains unaltered on disk.

---

## 5. Challenges & Trade-offs

- **Parameterized Tools vs Arbitrary Code Execution**: We prioritized safety and reproducibility by providing parameterized tool schemas over raw `eval()` execution.
- **Dual Engine Architecture**: Implemented a primary LLM integration (Google Gemini) alongside a local heuristic fallback engine for keyless offline evaluation.

---

## 6. Testing & Deployment Readiness

- **Verification**: Verified using a 7-test suite (`tests/test_agent.py`) covering product ranking, seasonality, regional comparisons, anomaly detection, schema queries, greetings, and multi-turn context.
- **Deployment**: Configured with relative dataset paths and environment variables (`GEMINI_API_KEY`) for instant deployment on Streamlit Community Cloud.
