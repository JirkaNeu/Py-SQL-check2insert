# Py-SQL-check2insert

checking for similarity using Chroma DB and LLMs before updating database via SQL 

This script:
* reads new entries from an excel-sheet to be inserted into a database
* reads already existing database entries
* uses Chroma DB and LLMs to determine the distance between each new entry (names of customers) and existing entries
* updates database with new entries when distance is greater than threshold
* asks the user what to do when entries have a smaller distance
* writes a basic log file in excel

