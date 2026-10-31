import pandas as pd
from agent.brain import ask
from tools.profiler import profile_dataframe, profile_to_text

df = pd.read_csv("data/titanic.csv")
summary = profile_to_text(profile_dataframe(df))

print(ask(f"Here is a dataset summary:\n{summary}\n\nGive me 3 interesting questions to explore."))