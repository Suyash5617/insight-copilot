"""
Insight Copilot - Streamlit Web Application UI
Natural Language Data Analyst Chatbot for SuperStore Sales Dataset
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os

from agent.graph import insight_graph
from agent.tools import get_schema_and_summary

# Streamlit Page Config
st.set_page_config(
    page_title="Insight Copilot | LangGraph Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .plan-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .number-card {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .matters-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 6px;
        padding: 12px 16px;
        color: #166534;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Multi-turn Session State
if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# Sidebar: Dataset / Schema Overview
# ==========================================
with st.sidebar:
    st.markdown("### 🧠 Insight Copilot")
    st.caption("LangGraph StateGraph Analyst Agent")
    st.markdown("---")

    st.markdown("#### 📋 Dataset Information")
    schema_res = get_schema_and_summary()

    if schema_res.get("status") == "success":
        sdata = schema_res["data"]
        st.success(f"Loaded: `{sdata['file_name']}`")
        st.metric("Total Records", f"{sdata['total_rows']:,}")
        
        col1, col2 = st.columns(2)
        col1.metric("Regions", len(sdata['regions']))
        col2.metric("Categories", len(sdata['categories']))
        
        st.markdown("**Core KPIs:**")
        kpis = sdata["kpis"]
        st.write(f"• **Total Sales**: ${kpis['total_sales']:,.2f}")
        st.write(f"• **Total Profit**: ${kpis['total_profit']:,.2f}")
        st.write(f"• **Total Orders**: {kpis['total_orders']:,}")

        with st.expander("🔍 View All Column Names"):
            st.write(sdata["columns"])
    else:
        st.error(f"Dataset Loading Error: {schema_res.get('message')}")

    st.markdown("---")
    st.markdown("#### 💡 Example Questions")
    sample_queries = [
        "What were the top 5 products by sales?",
        "Is there a seasonal trend in sales?",
        "Compare West and East region performance.",
        "Summarize anything unusual in this data.",
        "What columns are available in the dataset?"
    ]

    for sq in sample_queries:
        if st.button(f"👉 {sq}", use_container_width=True):
            st.session_state.selected_sample = sq

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# Main Header
# ==========================================
st.markdown('<div class="main-title">Insight Copilot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Reasoning Analyst Chatbot for SuperStore Sales Dataset (LangGraph StateGraph)</div>', unsafe_allow_html=True)


# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Show Agent Plan if present
        if msg.get("reasoning_plan"):
            with st.expander("📋 Agent Plan", expanded=False):
                st.markdown(msg["reasoning_plan"])
                if msg.get("selected_tool"):
                    st.caption(f"**Selected Tool**: `{msg['selected_tool']}`")

        st.markdown(msg["content"])

        # Render Chart if present
        if msg.get("visualization"):
            cspec = msg["visualization"]
            cdf = pd.DataFrame(cspec.get("data", []))
            if not cdf.empty:
                if cspec.get("chart_type") == "bar":
                    fig = px.bar(cdf, x=cspec.get("x_axis"), y=cspec.get("y_axis"), title=cspec.get("title", ""))
                elif cspec.get("chart_type") == "line":
                    fig = px.line(cdf, x=cspec.get("x_axis"), y=cspec.get("y_axis"), title=cspec.get("title", ""))
                else:
                    fig = px.scatter(cdf, x=cspec.get("x_axis"), y=cspec.get("y_axis"), title=cspec.get("title", ""))
                st.plotly_chart(fig, use_container_width=True, key=f"hist_chart_{idx}")


# ==========================================
# Input Box & Interaction
# ==========================================
user_prompt = st.chat_input("Ask a question about the SuperStore dataset...")

if "selected_sample" in st.session_state:
    user_prompt = st.session_state.pop("selected_sample")

if user_prompt:
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Process with LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("Processing with LangGraph Agent..."):
            try:
                initial_state = {
                    "user_query": user_prompt,
                    "messages": st.session_state.messages,
                    "reasoning_plan": None,
                    "requires_tool": True,
                    "selected_tool": None,
                    "tool_arguments": None,
                    "tool_result": None,
                    "visualization": None,
                    "final_insight": None,
                    "error": None
                }

                final_state = insight_graph.invoke(initial_state)

                plan = final_state.get("reasoning_plan", "")
                selected_tool = final_state.get("selected_tool")
                insight = final_state.get("final_insight", {})
                tool_result = final_state.get("tool_result", {})
                viz_spec = final_state.get("visualization")

                # Display Agent Plan
                if plan:
                    with st.expander("📋 Agent Plan", expanded=True):
                        st.markdown(plan)
                        if selected_tool:
                            st.caption(f"**Selected Tool**: `{selected_tool}`")

                # Display Analyst Answer Components
                st.markdown("### Answer")
                st.markdown(insight.get("direct_answer", ""))

                st.markdown("### Supporting Numbers")
                numbers = insight.get("supporting_numbers", [])
                if numbers:
                    for num in numbers:
                        st.markdown(f"- {num}")

                st.markdown("### Why This Matters")
                matters = insight.get("why_this_matters", "")
                st.markdown(f'<div class="matters-card">💡 {matters}</div>', unsafe_allow_html=True)

                # Render Auto-Visualizations
                curr_msg_idx = len(st.session_state.messages)
                if selected_tool == "query_dataset" and tool_result.get("status") == "success":
                    data_list = tool_result.get("data", [])
                    if data_list:
                        df_chart = pd.DataFrame(data_list)
                        dim_cols = [c for c in df_chart.columns if c not in ["Sales", "Profit", "Quantity", "Order_Count"]]
                        if dim_cols and "Sales" in df_chart.columns:
                            fig = px.bar(df_chart, x=dim_cols[0], y="Sales", color="Profit" if "Profit" in df_chart.columns else None, title="Sales & Profit Summary")
                            st.plotly_chart(fig, use_container_width=True, key=f"live_chart_query_{curr_msg_idx}")
                            viz_spec = {"chart_type": "bar", "x_axis": dim_cols[0], "y_axis": "Sales", "title": "Sales Summary", "data": data_list}

                elif viz_spec:
                    cdf = pd.DataFrame(viz_spec.get("data", []))
                    if not cdf.empty:
                        fig = px.bar(cdf, x=viz_spec.get("x_axis"), y=viz_spec.get("y_axis"), title=viz_spec.get("title", ""))
                        st.plotly_chart(fig, use_container_width=True, key=f"live_chart_viz_{curr_msg_idx}")

                # Save Assistant Response to History
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"### Answer\n{insight.get('direct_answer', '')}\n\n### Supporting Numbers\n" + "\n".join([f"- {n}" for n in numbers]) + f"\n\n### Why This Matters\n{matters}",
                    "reasoning_plan": plan,
                    "selected_tool": selected_tool,
                    "visualization": viz_spec
                })

            except Exception as e:
                st.error(f"An unexpected error occurred during execution: {str(e)}")
