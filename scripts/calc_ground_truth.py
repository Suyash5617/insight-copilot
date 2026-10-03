"""
Ground Truth Calculation Script for SuperStore_Sales_Dataset.csv
Calculates exact Pandas metrics directly from the real CSV file.
"""

import os
import pandas as pd
import numpy as np

csv_path = os.path.join("data", "SuperStore_Sales_Dataset.csv")
df = pd.read_csv(csv_path, encoding='utf-8')

# Parse dates with dayfirst=True
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce', dayfirst=True)
df['Year'] = df['Order Date'].dt.year
df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)

print("=== GROUND TRUTH CALCULATION REPORT ===")

print("\n1. DATASET OVERVIEW:")
print(f"Row Count: {len(df)}")
print(f"Col Count: {len(df.columns)}")
print(f"Col Names: {list(df.columns)}")
print(f"Date Range: {df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}")
print(f"Regions Count: {df['Region'].nunique()} -> {df['Region'].unique().tolist()}")
print(f"Categories Count: {df['Category'].nunique()} -> {df['Category'].unique().tolist()}")
print(f"Total Sales: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")
print(f"Unique Order IDs: {df['Order ID'].nunique()}")

print("\n2. CATEGORY METRICS:")
cat_summary = df.groupby('Category').agg(Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum')).reset_index()
print(cat_summary.to_string(index=False))
highest_sales_cat = cat_summary.loc[cat_summary['Total_Sales'].idxmax()]
highest_profit_cat = cat_summary.loc[cat_summary['Total_Profit'].idxmax()]
print(f"Highest Sales Category: {highest_sales_cat['Category']} (${highest_sales_cat['Total_Sales']:,.2f})")
print(f"Highest Profit Category: {highest_profit_cat['Category']} (${highest_profit_cat['Total_Profit']:,.2f})")

print("\n3. REGION METRICS:")
reg_summary = df.groupby('Region').agg(Total_Sales=('Sales', 'sum'), Total_Profit=('Profit', 'sum')).reset_index()
print(reg_summary.to_string(index=False))
highest_sales_reg = reg_summary.loc[reg_summary['Total_Sales'].idxmax()]
highest_profit_reg = reg_summary.loc[reg_summary['Total_Profit'].idxmax()]
print(f"Highest Sales Region: {highest_sales_reg['Region']} (${highest_sales_reg['Total_Sales']:,.2f})")
print(f"Highest Profit Region: {highest_profit_reg['Region']} (${highest_profit_reg['Total_Profit']:,.2f})")

print("\n4. TOP 5 PRODUCTS BY SALES:")
top5 = df.groupby('Product Name')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False).head(5)
for idx, row in top5.iterrows():
    pname = row['Product Name']
    psales = row['Sales']
    print(f"  - {pname}: ${psales:,.2f}")

print("\n5. TIME / SEASONAL TREND:")
yearly = df.groupby('Year')['Sales'].sum().reset_index()
print("Yearly Sales:")
print(yearly.to_string(index=False))

monthly = df.groupby('YearMonth')['Sales'].sum().reset_index()
print("\nMonthly Sales Sample (First 5 & Last 5):")
print(monthly.head(5).to_string(index=False))
print("...")
print(monthly.tail(5).to_string(index=False))

print("\n6. FURNITURE SPECIFIC METRICS:")
furn = df[df['Category'] == 'Furniture']
print(f"Furniture Exact Total Sales: ${furn['Sales'].sum():,.2f}")
print(f"Furniture Exact Total Profit: ${furn['Profit'].sum():,.2f}")
print(f"Furniture Exact Average Sales: ${furn['Sales'].mean():,.2f}")
print(f"Furniture Exact Average Profit: ${furn['Profit'].mean():,.2f}")

print("\n7. STATISTICAL EXTREMES & ANOMALIES:")
print(f"Min Sales: {df['Sales'].min()} | Max Sales: {df['Sales'].max()}")
print(f"Min Profit: {df['Profit'].min()} | Max Profit: {df['Profit'].max()}")
sub_margin = df.groupby(['Category', 'Sub-Category']).agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum')).reset_index()
sub_margin['Margin_%'] = (sub_margin['Profit'] / sub_margin['Sales'] * 100).round(2)
print("Sub-Category Margins Summary:")
print(sub_margin.sort_values(by='Margin_%').to_string(index=False))
