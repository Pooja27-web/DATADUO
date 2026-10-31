from pathlib import Path
import pandas as pd
from tools.cleaning import clean_dataframe
from tools.analysis import auto_analyze

df = pd.read_csv("data/titanic.csv")
cleaned, _ = clean_dataframe(df)

insights, figures = auto_analyze(cleaned, target="Survived")

print("INSIGHTS:")
for line in insights:
    print(" -", line)

Path("outputs").mkdir(exist_ok=True)
for name, fig in figures.items():
    fig.write_html(f"outputs/{name}.html")

print(f"\nSaved {len(figures)} charts to the 'outputs' folder")