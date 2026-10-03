"""
Typed LangGraph AgentState Definition for Insight Copilot
"""

from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Typed state object carried through the LangGraph StateGraph workflow.
    """
    user_query: str
    messages: List[Dict[str, Any]]
    reasoning_plan: Optional[str]
    requires_tool: bool
    selected_tool: Optional[str]
    tool_arguments: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    visualization: Optional[Dict[str, Any]]
    final_insight: Optional[Dict[str, Any]]
    error: Optional[str]
