from jne import prinje
try:
  with open("jne.txt", "r", encoding="utf-8") as notes:
    getnotes = []
    for lines in notes:
      getnotes.append(lines.strip("\r\n"))
    notes.close()
    path = getnotes[0]
    path = '/'.join(path.split('\\'))
    dbpass = []
    for i in range(5):
        dbpass.append(getnotes[i+1])
except:
  path = ""
  try: import ctypes; ctypes.windll.user32.MessageBoxW(0, "check path...", "Python", 1)
  except: print("check path...")

#--- read new data
new_data = path + "new_data.xlsx"

import pandas as pd
get_new_data = pd.read_excel(new_data)
new_data_df = get_new_data['new'].apply(str)
new_data_lst = [str(row) for row in new_data_df]

print(new_data_df)
print(new_data_lst)

#--- read sql
import psycopg2
conn = psycopg2.connect(database=dbpass[0],
                        host=dbpass[1],
                        user=dbpass[2],
                        password=dbpass[3],
                        port=dbpass[4])

cursor = conn.cursor()
cursor.execute('SELECT * FROM table_jne')
entries = cursor.fetchall()
conn.close()

print("--------------")
for i in entries:
    print(i)


#--- check with chromadb

#--- update sql

#--- update csv


