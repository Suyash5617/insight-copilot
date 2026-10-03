"""
Analytical Tools for Insight Copilot (Real SuperStore Sales Dataset)
Operates directly on data/SuperStore_Sales_Dataset.csv
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Optional

DATASET_PATH = os.path.join("data", "SuperStore_Sales_Dataset.csv")

def load_data() -> pd.DataFrame:
    """
    Loads and preprocesses the real SuperStore Sales Dataset.
    Does NOT modify the original CSV file on disk.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Real dataset CSV not found at '{os.path.abspath(DATASET_PATH)}'. "
            "Please ensure 'data/SuperStore_Sales_Dataset.csv' exists."
        )
    
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    
    # Parse dates carefully
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce', dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
    
    # Ensure numeric columns are strictly float/int
    for col in ['Sales', 'Profit', 'Quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df


def get_schema_and_summary() -> Dict[str, Any]:
    """
    Tool 1: Dataset Schema & Summary Explorer.
    Returns column details, data types, total record count, date ranges, unique categorical dimensions, and KPIs.
    """
    try:
        df = load_data()
        date_valid = df['Order Date'].dropna()
        
        schema_info = {
            "file_name": "SuperStore_Sales_Dataset.csv",
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(val) for col, val in df.isna().sum().items() if val > 0},
            "regions": df['Region'].dropna().unique().tolist() if 'Region' in df.columns else [],
            "categories": df['Category'].dropna().unique().tolist() if 'Category' in df.columns else [],
            "sub_categories": df['Sub-Category'].dropna().unique().tolist() if 'Sub-Category' in df.columns else [],
            "date_range": {
                "min_date": date_valid.min().strftime('%Y-%m-%d') if not date_valid.empty else "N/A",
                "max_date": date_valid.max().strftime('%Y-%m-%d') if not date_valid.empty else "N/A"
            },
            "kpis": {
                "total_sales": round(float(df['Sales'].sum()), 2),
                "total_profit": round(float(df['Profit'].sum()), 2),
                "total_orders": int(df['Order ID'].nunique()) if 'Order ID' in df.columns else len(df),
                "avg_order_sales": round(float(df['Sales'].mean()), 2)
            }
        }
        return {"status": "success", "tool": "get_schema_and_summary", "data": schema_info}
    except Exception as e:
        return {"status": "error", "tool": "get_schema_and_summary", "message": str(e)}


def query_dataset(
    group_by: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "Sales",
    ascending: bool = False,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Tool 2: Data Query & Aggregation Tool (Pandas / SQL filtering & grouping).
    Filters data, groups by dimensions (e.g. Region, Category, Product Name, YearMonth),
    and computes aggregated metrics (sum, mean, count).
    """
    try:
        df = load_data()

        # Apply filters safely
        if filters:
            for col, val in filters.items():
                if col in df.columns:
                    if isinstance(val, list):
                        df = df[df[col].isin(val)]
                    else:
                        df = df[df[col] == val]

        # Default metrics
        if not metrics:
            metrics = ["Sales", "Profit", "Quantity"]
        metrics = [m for m in metrics if m in df.columns]

        # Grouping logic
        if group_by:
            group_cols = [g for g in group_by if g in df.columns]
            if not group_cols:
                group_cols = ["Category"]
            
            agg_dict = {m: 'sum' for m in metrics}
            if 'Order ID' in df.columns and 'Order ID' not in group_cols:
                agg_dict['Order ID'] = 'nunique'

            result_df = df.groupby(group_cols).agg(agg_dict).reset_index()
            if 'Order ID' in result_df.columns:
                result_df.rename(columns={'Order ID': 'Order_Count'}, inplace=True)
        else:
            result_df = df[metrics].agg(['sum', 'mean', 'min', 'max']).reset_index()

        # Sorting & limiting
        if sort_by in result_df.columns:
            result_df = result_df.sort_values(by=sort_by, ascending=ascending)
        elif 'Sales' in result_df.columns:
            result_df = result_df.sort_values(by='Sales', ascending=False)
            
        result_df = result_df.head(limit)

        # Round numerical output
        for col in result_df.select_dtypes(include=[np.number]).columns:
            result_df[col] = result_df[col].round(2)

        records = result_df.to_dict(orient="records")
        return {
            "status": "success",
            "tool": "query_dataset",
            "row_count": len(result_df),
            "data": records,
            "columns": list(result_df.columns)
        }
    except Exception as e:
        return {"status": "error", "tool": "query_dataset", "message": str(e)}


def calculate_statistics(
    analysis_type: str = "growth", # "growth", "regional_comparison", "anomaly", "summary_stats", "correlation"
    metric: str = "Sales",
    dimension: str = "Region"
) -> Dict[str, Any]:
    """
    Tool 3: Statistical Calculation Tool.
    Computes YoY growth, regional variance, profit margins, correlation, or anomaly detection.
    """
    try:
        df = load_data()

        if analysis_type in ["growth", "trend"]:
            # Year-over-year growth rate calculation
            yearly = df.groupby('Year')[metric].sum().reset_index()
            yearly = yearly[yearly['Year'].notna() & (yearly['Year'] > 2000)]
            yearly['YoY_Growth_%'] = yearly[metric].pct_change() * 100
            yearly['YoY_Growth_%'] = yearly['YoY_Growth_%'].round(2).fillna(0)
            yearly[metric] = yearly[metric].round(2)
            return {
                "status": "success",
                "tool": "calculate_statistics",
                "analysis_type": "growth",
                "data": yearly.to_dict(orient="records")
            }

        elif analysis_type == "regional_comparison":
            grouped = df.groupby('Region').agg(
                Total_Sales=('Sales', 'sum'),
                Total_Profit=('Profit', 'sum'),
                Avg_Profit=('Profit', 'mean'),
                Order_Count=('Order ID', 'nunique') if 'Order ID' in df.columns else ('Sales', 'count')
            ).reset_index()

            grouped['Profit_Margin_%'] = np.where(
                grouped['Total_Sales'] > 0,
                (grouped['Total_Profit'] / grouped['Total_Sales'] * 100).round(2),
                0
            )
            grouped['Total_Sales'] = grouped['Total_Sales'].round(2)
            grouped['Total_Profit'] = grouped['Total_Profit'].round(2)
            grouped['Avg_Profit'] = grouped['Avg_Profit'].round(2)

            return {
                "status": "success",
                "tool": "calculate_statistics",
                "analysis_type": "regional_comparison",
                "data": grouped.to_dict(orient="records")
            }

        elif analysis_type == "anomaly":
            # Outlier detection on sub-category profit margins
            sub_agg = df.groupby(['Category', 'Sub-Category']).agg(
                Total_Sales=('Sales', 'sum'),
                Total_Profit=('Profit', 'sum')
            ).reset_index()

            sub_agg['Profit_Margin_%'] = np.where(
                sub_agg['Total_Sales'] > 0,
                (sub_agg['Total_Profit'] / sub_agg['Total_Sales'] * 100).round(2),
                0
            )
            mean_margin = sub_agg['Profit_Margin_%'].mean()
            std_margin = sub_agg['Profit_Margin_%'].std()

            sub_agg['Z_Score'] = (sub_agg['Profit_Margin_%'] - mean_margin) / (std_margin if std_margin != 0 else 1)
            anomalies = sub_agg[(sub_agg['Z_Score'] > 1.2) | (sub_agg['Z_Score'] < -1.2)].copy()

            sub_agg['Total_Sales'] = sub_agg['Total_Sales'].round(2)
            sub_agg['Total_Profit'] = sub_agg['Total_Profit'].round(2)

            return {
                "status": "success",
                "tool": "calculate_statistics",
                "analysis_type": "anomaly",
                "overall_avg_margin": round(float(mean_margin), 2),
                "data": anomalies.to_dict(orient="records")
            }

        elif analysis_type == "correlation":
            numeric_df = df[['Sales', 'Profit', 'Quantity']].dropna()
            corr = numeric_df.corr().round(3).to_dict()
            return {
                "status": "success",
                "tool": "calculate_statistics",
                "analysis_type": "correlation",
                "data": corr
            }

        else: # Summary statistics
            stats = df[[metric, 'Profit', 'Quantity']].describe().round(2).to_dict()
            return {
                "status": "success",
                "tool": "calculate_statistics",
                "analysis_type": "summary_stats",
                "data": stats
            }

    except Exception as e:
        return {"status": "error", "tool": "calculate_statistics", "message": str(e)}


def generate_visualization(
    chart_type: str = "bar", # "bar", "line", "scatter", "pie"
    x_axis: str = "Category",
    y_axis: str = "Sales",
    color_by: Optional[str] = None,
    title: str = "SuperStore Sales Analysis"
) -> Dict[str, Any]:
    """
    Tool 4: Plotly Chart & Visualization Generator.
    Prepares dataset records and returns chart specification suitable for Plotly rendering in Streamlit.
    """
    try:
        df = load_data()

        if chart_type in ["bar", "pie"]:
            if x_axis in df.columns and y_axis in df.columns:
                chart_df = df.groupby(x_axis)[y_axis].sum().reset_index()
                chart_df = chart_df.sort_values(by=y_axis, ascending=False).head(15)
                chart_df[y_axis] = chart_df[y_axis].round(2)
            else:
                chart_df = df.groupby("Category")["Sales"].sum().reset_index()
                x_axis, y_axis = "Category", "Sales"

        elif chart_type == "line":
            df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
            x_axis = "YearMonth"
            if color_by and color_by in df.columns:
                chart_df = df.groupby(['YearMonth', color_by])[y_axis].sum().reset_index()
            else:
                chart_df = df.groupby('YearMonth')[y_axis].sum().reset_index()
            chart_df[y_axis] = chart_df[y_axis].round(2)

        else: # scatter
            chart_df = df.groupby("Sub-Category")[["Sales", "Profit"]].sum().reset_index()
            x_axis, y_axis = "Sales", "Profit"
            chart_df['Sales'] = chart_df['Sales'].round(2)
            chart_df['Profit'] = chart_df['Profit'].round(2)

        return {
            "status": "success",
            "tool": "generate_visualization",
            "chart_spec": {
                "chart_type": chart_type,
                "title": title,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "color_by": color_by,
                "data": chart_df.to_dict(orient="records")
            }
        }
    except Exception as e:
        return {"status": "error", "tool": "generate_visualization", "message": str(e)}
