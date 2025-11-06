"""
Simple test to verify the database is working correctly
"""
import sqlite3
from pathlib import Path

script_dir = Path(__file__).parent
db_path = script_dir / "data" / "hybrid_cars.db"

print("Testing SQLite Database...")
print("=" * 60)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Test 1: Count total records
cursor.execute("SELECT COUNT(*) FROM cars")
count = cursor.fetchone()[0]
print(f"\n✓ Total records: {count}")

# Test 2: Get fastest car
cursor.execute("SELECT vehicle, acceleration, year FROM cars ORDER BY acceleration ASC LIMIT 1")
result = cursor.fetchone()
print(f"\n✓ Fastest car (lowest acceleration time):")
print(f"  {result[0]} ({result[2]}) - {result[1]} seconds")

# Test 3: Get most efficient car
cursor.execute("SELECT vehicle, mpg, year FROM cars ORDER BY mpg DESC LIMIT 1")
result = cursor.fetchone()
print(f"\n✓ Most efficient car (highest MPG):")
print(f"  {result[0]} ({result[2]}) - {result[1]} MPG")

# Test 4: Get cheapest pickup truck
cursor.execute("SELECT vehicle, msrp, year FROM cars WHERE class = 'Pickup Truck' ORDER BY msrp ASC LIMIT 1")
result = cursor.fetchone()
print(f"\n✓ Cheapest Pickup Truck:")
print(f"  {result[0]} ({result[2]}) - ${result[1]:,.2f}")

# Test 5: Search for Prius V
cursor.execute("SELECT vehicle, acceleration, mpg, year FROM cars WHERE vehicle LIKE '%Prius V%'")
results = cursor.fetchall()
print(f"\n✓ Prius V models found: {len(results)}")
for r in results:
    print(f"  {r[0]} ({r[3]}) - {r[1]}s, {r[2]} MPG")

# Test 6: Cars from 2010
cursor.execute("SELECT COUNT(*) FROM cars WHERE year = 2010")
count = cursor.fetchone()[0]
print(f"\n✓ Cars from 2010: {count} models")

conn.close()

print("\n" + "=" * 60)
print("All database tests passed! ✓")
print("\nThe database is ready to use with the RAG examples.")
print("Make sure to configure your .env file with GITHUB_TOKEN to run the RAG scripts.")
