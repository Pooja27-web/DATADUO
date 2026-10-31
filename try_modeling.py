from pathlib import Path
import pandas as pd
from tools.cleaning import clean_dataframe
from tools.modeling import auto_model

df = pd.read_csv("data/titanic.csv")
cleaned, _ = clean_dataframe(df)

out = auto_model(cleaned, target="Survived")

print("Task:", out["task"], "| metric:", out["metric"])
print("Dropped columns:", out["dropped_columns"])
print("\nMODEL COMPARISON (5-fold cross-validation):")
print(out["results"].round(3).to_string(index=False))
print("\nBest model:", out["best_model"])
print(f"\nTOP FEATURES ({out['importance_method']}):")
print(out["importance"].head(8).round(3).to_string())

Path("outputs").mkdir(exist_ok=True)
out["figure"].write_html("outputs/feature_importance.html")