import pandas as pd
from tools.cleaning import clean_dataframe
from tools.profiler import profile_dataframe, profile_to_text

df = pd.read_csv("data/titanic.csv")
cleaned, log = clean_dataframe(df)

print("ACTIONS TAKEN:")
for line in log:
    print(" -", line)

print("\nMISSING VALUES LEFT:", int(cleaned.isna().sum().sum()))
print("SHAPE:", df.shape, "->", cleaned.shape)
print("\nNEW PROFILE:")
print(profile_to_text(profile_dataframe(cleaned)))
