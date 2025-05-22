from jne import prinje

try:
  with open("jne.txt", "r", encoding="utf-8") as notes:
    getnotes = []
    for lines in notes:
      getnotes.append(lines.strip("\r\n"))
    notes.close()
    path = getnotes[0]
    path = '/'.join(path.split('\\'))
except:
  path = ""
  try: import ctypes; ctypes.windll.user32.MessageBoxW(0, "check path...", "Python", 1)
  except: print("check path...")


#--- read sql

#--- read new data
new_data = path + "new_data.xlsx"

import pandas as pd
get_new_data = pd.read_excel(new_data)
col_data = get_new_data['new'].apply(str)
control_data_jne = [str(row) for row in col_data]

print(control_data_jne)
print(col_data)

#--- check with chromadb

#--- update sql

#--- update csv


