"""
Verification and Test Suite for Insight Copilot (Real SuperStore Sales Dataset)
Tests the 7 mandatory assignment questions and LangGraph state routing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import insight_graph

def run_tests():
    print("==================================================")
    print("INSIGHT COPILOT — AGENT & LANGGRAPH TEST SUITE")
    print("==================================================\n")

    test_cases = [
        {
            "id": 1,
            "query": "What were the top 5 products by sales?",
            "expected_tool": "query_dataset"
        },
        {
            "id": 2,
            "query": "Is there a seasonal trend in sales?",
            "expected_tool": "calculate_statistics"
        },
        {
            "id": 3,
            "query": "Compare West and East region performance and explain what's driving the difference.",
            "expected_tool": "calculate_statistics"
        },
        {
            "id": 4,
            "query": "Summarize anything unusual in this data.",
            "expected_tool": "calculate_statistics"
        },
        {
            "id": 5,
            "query": "What columns are available in the dataset?",
            "expected_tool": "get_schema_and_summary"
        },
        {
            "id": 6,
            "query": "Hello, who are you?",
            "expected_tool": None # direct_responder
        },
        {
            "id": 7,
            "query": "What about total profit for category Furniture?",
            "expected_tool": "query_dataset"
        }
    ]

    passed = 0
    for tc in test_cases:
        print(f"Test #{tc['id']}: Query -> '{tc['query']}'")
        res = insight_graph.invoke({
            "user_query": tc['query'],
            "messages": [],
            "reasoning_plan": None,
            "requires_tool": True,
            "selected_tool": None,
            "tool_arguments": None,
            "tool_result": None,
            "visualization": None,
            "final_insight": None,
            "error": None
        })

        selected = res.get("selected_tool")
        plan = res.get("reasoning_plan", "")
        insight = res.get("final_insight", {})

        print(f"  • Plan: {plan.splitlines()[0] if plan else 'None'}")
        print(f"  • Selected Tool: '{selected}' (Expected: '{tc['expected_tool']}')")
        print(f"  • Direct Answer: {insight.get('direct_answer')[:120]}...")
        print(f"  • Supporting Numbers: {insight.get('supporting_numbers')[:2]}")
        print(f"  • Why This Matters: {insight.get('why_this_matters')[:100]}")

        if selected == tc['expected_tool']:
            print("  [PASS] STATUS: PASSED\n")
            passed += 1
        else:
            print(f"  [FAIL] STATUS: FAILED (Got '{selected}', expected '{tc['expected_tool']}')\n")

    print("==================================================")
    print(f"TEST RESULTS: {passed} / {len(test_cases)} PASSED")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
