"""
LangGraph StateGraph Definition for Insight Copilot
Nodes: planner, tool_runner, synthesizer, direct_responder
Routing: Real conditional router edge (route_after_planner)
"""

from typing import Dict, Any, Literal
from agent.state import AgentState
from agent.tools import (
    query_dataset,
    calculate_statistics,
    generate_visualization,
    get_schema_and_summary
)
from agent.llm import generate_plan_and_tool_decision, synthesize_insight

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ==========================================
# Graph Nodes
# ==========================================

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Planner Node
    Articulates a short user-facing plan/rationale (WHAT the agent plans to do),
    evaluates if dataset analysis is required, and selects tool + arguments.
    """
    user_query = state.get("user_query", "")
    history = state.get("messages", [])

    plan, requires_tool, selected_tool, tool_arguments = generate_plan_and_tool_decision(user_query, history)

    return {
        "reasoning_plan": plan,
        "requires_tool": requires_tool,
        "selected_tool": selected_tool,
        "tool_arguments": tool_arguments
    }


def route_after_planner(state: AgentState) -> Literal["tool_runner", "direct_responder"]:
    """
    Conditional Router Edge:
    Inspects state to route to tool_runner (for dataset analysis) or direct_responder.
    """
    if state.get("requires_tool", False):
        return "tool_runner"
    return "direct_responder"


def tool_runner_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Tool Runner Node
    Executes the selected analytical tool using Pandas/Plotly engines on the real CSV.
    """
    tool_name = state.get("selected_tool")
    tool_args = state.get("tool_arguments") or {}

    tool_result = None
    visualization_spec = None

    if tool_name == "query_dataset":
        tool_result = query_dataset(**tool_args)
    elif tool_name == "calculate_statistics":
        tool_result = calculate_statistics(**tool_args)
    elif tool_name == "generate_visualization":
        tool_result = generate_visualization(**tool_args)
        if tool_result.get("status") == "success":
            visualization_spec = tool_result.get("chart_spec")
    elif tool_name == "get_schema_and_summary":
        tool_result = get_schema_and_summary()
    else:
        tool_result = {"status": "error", "message": f"Tool '{tool_name}' is not recognized."}

    return {
        "tool_result": tool_result,
        "visualization": visualization_spec
    }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Synthesizer Node
    Transforms raw tool output + plan into a structured analyst response:
    - Answer
    - Supporting Numbers
    - Why This Matters
    """
    user_query = state.get("user_query", "")
    plan = state.get("reasoning_plan", "")
    tool_name = state.get("selected_tool", "")
    tool_result = state.get("tool_result") or {}

    insight = synthesize_insight(user_query, plan, tool_name, tool_result)
    return {"final_insight": insight}


def direct_responder_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Direct Responder Node
    Handles questions that do not require dataset analysis (greetings, out-of-scope prompts).
    """
    user_query = state.get("user_query", "").lower()

    if any(w in user_query for w in ["weather", "president", "recipe", "movie", "football", "stock"]):
        answer_text = "I don't have a tool or data source available to answer that question. I am specialized in analyzing the SuperStore Sales Dataset."
    else:
        answer_text = (
            "I’m **Insight Copilot**, an AI assistant that analyzes the Global SuperStore Sales Dataset "
            "and explains business insights.\n\n"
            "You can ask me questions like:\n"
            "• *'What were the top 5 products by sales?'*\n"
            "• *'Compare West and East region performance.'*\n"
            "• *'Is there a seasonal trend in sales?'*\n"
            "• *'Summarize anything unusual in this data.'*\n"
            "• *'What columns are available in the dataset?'*"
        )

    insight = {
        "direct_answer": answer_text,
        "supporting_numbers": ["N/A - Non-analytical query"],
        "why_this_matters": "Insight Copilot uses LangGraph conditional routing to answer conversational prompts without executing unnecessary data tools."
    }
    return {"final_insight": insight}


# ==========================================
# LangGraph Workflow Construction
# ==========================================

def build_insight_graph():
    """Builds and compiles the real LangGraph StateGraph."""
    if LANGGRAPH_AVAILABLE:
        builder = StateGraph(AgentState)

        # Add Nodes
        builder.add_node("planner", planner_node)
        builder.add_node("tool_runner", tool_runner_node)
        builder.add_node("synthesizer", synthesizer_node)
        builder.add_node("direct_responder", direct_responder_node)

        # Add Edges
        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", route_after_planner, {
            "tool_runner": "tool_runner",
            "direct_responder": "direct_responder"
        })
        builder.add_edge("tool_runner", "synthesizer")
        builder.add_edge("synthesizer", END)
        builder.add_edge("direct_responder", END)

        return builder.compile()
    else:
        # Fallback executor adhering strictly to the same LangGraph state transition contract
        class FallbackGraph:
            def invoke(self, initial_state: AgentState) -> AgentState:
                state = dict(initial_state)
                state.update(planner_node(state))
                next_step = route_after_planner(state)
                if next_step == "tool_runner":
                    state.update(tool_runner_node(state))
                    state.update(synthesizer_node(state))
                else:
                    state.update(direct_responder_node(state))
                return state

        return FallbackGraph()

insight_graph = build_insight_graph()
