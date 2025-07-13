from jne import prinje
from jne import loogress
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

print("new_data_df:")
print(new_data_df)
print("new_data_lst:")
print(new_data_lst)

#--- read sql
import psycopg2
conn = psycopg2.connect(database=dbpass[0],
                        host=dbpass[1],
                        user=dbpass[2],
                        password=dbpass[3],
                        port=dbpass[4])

cur = conn.cursor()

#cur.execute('SELECT * FROM bands')
cur.execute('SELECT name FROM bands')
sql_data = cur.fetchall()
#conn.commit()
cur.close()
conn.close()


sql_data_lst = []
for entries in sql_data:
    sql_data_lst.append(entries[0])

print("--------------")
print("sql data received:")
print(sql_data_lst)
print("--------------\n")


#--- check with chromadb

import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
client = chromadb.Client()
#dbPath = path + "chroma"
#client = chromadb.PersistentClient(path=dbPath)

import torch
from transformers import AutoTokenizer, AutoModel
#model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-de', trust_remote_code=True, torch_dtype=torch.bfloat16)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/LaBSE')
#model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

class Embedding_Function(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = model.encode(input)
        return(embeddings.tolist())

my_embeddings = Embedding_Function()
dbName = "SQLvecDB"
dbDocs = sql_data_lst

#collection = client.get_or_create_collection(
collection = client.create_collection(
    name=dbName,
    embedding_function=my_embeddings,
    metadata={"hnsw:space": "cosine"}
)

for i in range(len(dbDocs)):
    collection.add(
        documents=dbDocs[i],
        ids=str(i),
        metadatas={"source": "sql_data"}
    )
    exec('try:loogress(i, len(dbDocs))\nexcept:print("thinking...")')


print("\nlen new data:")
print(len(new_data_lst))
print("")


dist_threshold = 0.45
results = []
results_2check = []

for i in range(len(new_data_lst)):
    db_query = collection.query(query_texts=[new_data_lst[i]], n_results=len(dbDocs))
    nearest_embeddings = db_query['ids'][0]
    embedding_document = db_query['documents'][0]
    distances = db_query['distances'][0]
    filtered_results = [(id, doc, dist) for id, doc, dist in zip(nearest_embeddings, embedding_document, distances) if dist <= dist_threshold]
    #filtered_results = [(id, dist) for id, dist in zip(nearest_embeddings, distances) if dist <= dist_threshold]
    #print(new_data_lst[i])
    #print(filtered_results)
    #print(len(filtered_results))
    for x in range(len(filtered_results)):
        print(f"new data 2check: {new_data_lst[i]} ---> similar to: {filtered_results[x][1]}")
        results_2check.append(new_data_lst[i])
    results.append(filtered_results)


print("---------")
#print(len(results))
#print(results)
print(results_2check)

#create list with new entries where dist. > dist_threshold
print("The following data will be sent to the database:")

ready4db = list(set(new_data_lst) - set(results_2check))
print(ready4db)

#------- print csv as log-file -------#
#-------------------------------------#
import os
import csv

def fill_file(file_name):
    with open(file_name, 'a', newline='') as file:
        fill_file = csv.writer(file)
        fill_file.writerows(logfile_list)
    return None

def results_to_file():
    file_name = path + "new_data_log.csv"
    if not os.path.exists(file_name):
        head_row = [['new data', 'status']]
        with open(file_name, 'w', newline='') as file:
            preparefile = csv.writer(file)
            preparefile.writerows(head_row)
        fill_file(file_name)
    else:
        fill_file(file_name)
    return None

logfile_list = [(item, "updated to database") for item in ready4db]

print("logfile_list")
print(logfile_list)
results_to_file()



#ask for each with dist < 45


#--- update sql




