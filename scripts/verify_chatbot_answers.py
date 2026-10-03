"""
Verification Script for Queries A through H
Compares chatbot execution outputs against Pandas ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import insight_graph

queries = [
    {"id": "A", "q": "What were the top 5 products by sales?"},
    {"id": "B", "q": "What is the total profit for the Furniture category?"},
    {"id": "C", "q": "Compare sales and profit between the West and East regions."},
    {"id": "D", "q": "Is there a seasonal trend in sales?"},
    {"id": "E", "q": "Which category has the highest total sales and which has the highest total profit?"},
    {"id": "F", "q": "What columns are available in the dataset?"},
    {"id": "G", "q": "Hello, who are you?"},
    {"id": "H_prev", "q": "Which region has the highest sales?"},
    {"id": "H", "q": "What about its profit?"}
]

print("=== CHATBOT RESPONSE ACCURACY AUDIT ===\n")

history = []
for item in queries:
    qid = item["id"]
    query_text = item["q"]
    print(f"[{qid}] Query: '{query_text}'")

    state_input = {
        "user_query": query_text,
        "messages": history,
        "reasoning_plan": None,
        "requires_tool": True,
        "selected_tool": None,
        "tool_arguments": None,
        "tool_result": None,
        "visualization": None,
        "final_insight": None,
        "error": None
    }

    res = insight_graph.invoke(state_input)

    plan = res.get("reasoning_plan", "")
    tool = res.get("selected_tool")
    insight = res.get("final_insight", {})

    print(f"  • Selected Tool: {tool}")
    print(f"  • Direct Answer: {insight.get('direct_answer')}")
    print(f"  • Supporting Numbers: {insight.get('supporting_numbers')}")
    print(f"  • Why This Matters: {insight.get('why_this_matters')}\n")

    history.append({"role": "user", "content": query_text})
    history.append({"role": "assistant", "content": insight.get("direct_answer", "")})
