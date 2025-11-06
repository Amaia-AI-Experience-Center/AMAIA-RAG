# Exercise 2: RAG with SQL Database

This exercise demonstrates using RAG (Retrieval-Augmented Generation) with a SQL database instead of CSV files.

## What's New?

Instead of using a CSV file with lunr search, this exercise:
1. Uses a **SQLite database** with structured car data
2. Uses an **LLM to generate SQL queries** from natural language questions (Text-to-SQL)
3. Executes the queries and uses results to answer questions

## Files

- `create_database.py` - Creates the SQLite database from the CSV data
- `rag_sql.py` - Simple RAG with SQL query generation
- `data/hybrid_cars.db` - SQLite database with hybrid/electric car data
- `data/hybrid.csv` - Original CSV data (kept for reference)

## Setup

1. Make sure you have a `.env` file in the root directory with your `GITHUB_TOKEN`:
   ```bash
   cp ../.env.example ../.env
   # Edit .env and add your GitHub token
   ```

2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Create the database (if not already created):
   ```bash
   python create_database.py
   ```

## Running the Example

```bash
python rag_sql.py
```

This will:
1. Connect to the SQLite database
2. Use the LLM to generate a SQL query from the question "how fast is the prius v?"
3. Execute the query
4. Use the results to generate a natural language answer

## Database Schema

```sql
Table: cars
Columns:
  - id (INTEGER): Primary key
  - vehicle (TEXT): Name of the car model
  - year (INTEGER): Year of manufacture
  - msrp (REAL): Manufacturer's Suggested Retail Price in dollars
  - acceleration (REAL): 0-60 mph time in seconds (lower is faster)
  - mpg (REAL): Miles per gallon (higher is more efficient)
  - class (TEXT): Vehicle class (Compact, Midsize, SUV, Pickup Truck, etc.)
```

## How Text-to-SQL Works

1. **Schema Context**: The LLM is given the database schema
2. **Query Generation**: The LLM converts natural language to SQL
3. **Query Execution**: The SQL query is executed against the database
4. **Answer Generation**: The LLM uses the query results to answer the question

## Example Questions

- "how fast is the prius v?"
- "what is the fastest car?"
- "show me efficient cars"
- "what's the cheapest pickup truck?"
- "cars from 2010"

## Advantages of SQL over CSV

1. **Structure**: Better data organization and relationships
2. **Querying**: More powerful query capabilities (JOIN, GROUP BY, etc.)
3. **Performance**: Better for larger datasets
4. **Accuracy**: Structured queries are more precise than keyword search
5. **Complex Questions**: Can answer complex analytical questions
