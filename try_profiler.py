import pandas as pd
from tools.profiler import profile_dataframe, profile_to_text

df = pd.read_csv("data/titanic.csv")
print(profile_to_text(profile_dataframe(df)))