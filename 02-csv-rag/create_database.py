"""
Script to create SQLite database from hybrid cars CSV
"""
import sqlite3
import csv
from pathlib import Path

# Path to the folder where this script lives
script_dir = Path(__file__).parent

# Path to the CSV inside the 'data' folder
csv_path = script_dir / "data" / "hybrid.csv"

# Path for the SQLite database
db_path = script_dir / "data" / "hybrid_cars.db"

# Remove existing database if it exists
if db_path.exists():
    db_path.unlink()

# Create database and table
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table with appropriate schema
cursor.execute("""
CREATE TABLE cars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle TEXT NOT NULL,
    year INTEGER,
    msrp REAL,
    acceleration REAL,
    mpg REAL,
    class TEXT
)
""")

# Read CSV and insert data
with open(csv_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Skip empty rows
        if not row['vehicle'].strip():
            continue

        cursor.execute("""
            INSERT INTO cars (vehicle, year, msrp, acceleration, mpg, class)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row['vehicle'],
            int(row['year']) if row['year'] else None,
            float(row['msrp']) if row['msrp'] else None,
            float(row['acceleration']) if row['acceleration'] else None,
            float(row['mpg']) if row['mpg'] else None,
            row['class']
        ))

conn.commit()

# Verify data was inserted
cursor.execute("SELECT COUNT(*) FROM cars")
count = cursor.fetchone()[0]
print(f"Database created successfully at: {db_path}")
print(f"Total records inserted: {count}")

# Show some sample data
cursor.execute("SELECT * FROM cars LIMIT 5")
print("\nSample data:")
for row in cursor.fetchall():
    print(row)

# Show schema
cursor.execute("PRAGMA table_info(cars)")
print("\nTable schema:")
for col in cursor.fetchall():
    print(f"  {col[1]} ({col[2]})")

conn.close()
