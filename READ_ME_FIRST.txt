DATADUO redesign. Extract this zip INTO your DATADUO project folder (the one that contains app.py)
and choose "Replace the files in the destination". Do NOT replace the whole folder.

New files:      tools/palette.py, tools/background.py, .streamlit/config.toml
Replaced files: app.py, tools/theme.py, tools/charts.py, tools/business.py, tools/deck.py,
                tools/report.py, tools/loaders.py, make_sample_sales.py

One manual edit (2 places) in tools/analysis.py:
  VS Code: Ctrl + H, find   "RdBu_r"
                    replace ["#E8604C", "#161616", "#7FD1A0"]
  then Replace All.

Then:
  pip install pillow openpyxl xlrd pyarrow
  python make_sample_sales.py
  python -m streamlit run app.py
