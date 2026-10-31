"""Creates data/sample_sales.csv: a realistic FAKE sales dataset for trying the business dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 6000

# category, product, price, base profit margin
products = [
    ("Technology", "Phone X", 700, 0.22), ("Technology", "Laptop Pro", 1400, 0.15),
    ("Technology", "Tablet Air", 500, 0.20), ("Technology", "Headphones", 120, 0.35),
    ("Technology", "Smart Watch", 250, 0.28), ("Technology", "Monitor 27", 320, 0.18),
    ("Furniture", "Office Chair", 220, 0.12), ("Furniture", "Standing Desk", 480, 0.10),
    ("Furniture", "Bookcase", 260, 0.02), ("Furniture", "Sofa", 900, 0.08),
    ("Furniture", "Coffee Table", 340, -0.04), ("Furniture", "Filing Cabinet", 180, 0.14),
    ("Office Supplies", "Paper Pack", 25, 0.40), ("Office Supplies", "Binder Set", 18, 0.38),
    ("Office Supplies", "Stapler", 15, 0.30), ("Office Supplies", "Pens Box", 12, 0.45),
    ("Office Supplies", "Labels", 9, 0.42), ("Office Supplies", "Notebook", 8, 0.35),
]
# state: (region, sales weight)
states = {
    "California": ("West", 0.20), "Washington": ("West", 0.06), "Colorado": ("West", 0.04),
    "Texas": ("South", 0.12), "Florida": ("South", 0.08), "Georgia": ("South", 0.05),
    "Virginia": ("South", 0.04), "New York": ("East", 0.16), "Pennsylvania": ("East", 0.06),
    "Illinois": ("Central", 0.08), "Ohio": ("Central", 0.06), "Michigan": ("Central", 0.05),
}

# dates: 2 years, growing over time, strong Nov/Dec
days = pd.date_range("2023-01-01", "2024-12-31")
season = np.where(days.month.isin([11, 12]), 1.7, np.where(days.month.isin([3, 9]), 1.2, 1.0))
w = season * (1 + 0.6 * np.arange(len(days)) / len(days))
order_dates = pd.to_datetime(rng.choice(days.values, size=N, p=w / w.sum()))

# customers: a few big ones, a long tail of small ones
n_cust = 400
cw = 1 / np.arange(1, n_cust + 1) ** 0.7
cust = rng.choice(n_cust, size=N, p=cw / cw.sum())
seg_of = rng.choice(["Consumer", "Corporate", "Home Office"], size=n_cust, p=[0.55, 0.30, 0.15])

pidx = rng.choice(len(products), size=N, p=rng.dirichlet(np.ones(len(products)) * 3))
state_names = list(states)
sw = np.array([states[s][1] for s in state_names])
st = rng.choice(state_names, size=N, p=sw / sw.sum())

category = np.array([products[i][0] for i in pidx])
price = np.array([products[i][2] for i in pidx]) * rng.normal(1, 0.05, N)
margin = np.array([products[i][3] for i in pidx])
qty = rng.integers(1, 8, N)
disc = rng.choice([0, 0, 0, 0.1, 0.2, 0.3, 0.4], size=N)
disc = np.where(category == "Furniture", rng.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], size=N), disc)
sales = price * qty * (1 - disc)
profit = sales * (margin - 0.55 * disc + rng.normal(0, 0.03, N))

df = pd.DataFrame({
    "Order ID": [f"ORD-{100000 + i}" for i in range(N)],
    "Order Date": order_dates.strftime("%Y-%m-%d"),
    "Customer Name": [f"Customer {c + 1:03d}" for c in cust],
    "Segment": seg_of[cust],
    "Product Name": [products[i][1] for i in pidx],
    "Category": category,
    "Region": [states[s][0] for s in st],
    "State": st,
    "Country": "United States",
    "Quantity": qty,
    "Discount": disc,
    "Sales": sales.round(2),
    "Profit": profit.round(2),
}).sort_values("Order Date")

Path("data").mkdir(exist_ok=True)
df.to_csv("data/sample_sales.csv", index=False)
print("Saved data/sample_sales.csv", df.shape)
