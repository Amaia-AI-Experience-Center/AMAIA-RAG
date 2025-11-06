"""
Interactive RAG with SQL - Text-to-SQL Query Generation with Conversation History
This example demonstrates using an LLM to generate SQL queries from natural language,
maintaining conversation context.
"""
import sqlite3
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

# Path to the folder where this script lives
script_dir = Path(__file__).parent

# Path to the SQLite database (using the one from exercise 2)
db_path = script_dir.parent / "02-csv-rag" / "data" / "hybrid_cars.db"

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get database schema for the LLM
cursor.execute("PRAGMA table_info(cars)")
schema_info = cursor.fetchall()
schema_description = "Table: cars\nColumns:\n"
for col in schema_info:
    schema_description += f"  - {col[1]} ({col[2]}): "
    if col[1] == "vehicle":
        schema_description += "Name of the car model"
    elif col[1] == "year":
        schema_description += "Year of manufacture"
    elif col[1] == "msrp":
        schema_description += "Manufacturer's Suggested Retail Price in dollars"
    elif col[1] == "acceleration":
        schema_description += "0-60 mph time in seconds (lower is faster)"
    elif col[1] == "mpg":
        schema_description += "Miles per gallon (higher is more efficient)"
    elif col[1] == "class":
        schema_description += "Vehicle class (Compact, Midsize, SUV, etc.)"
    schema_description += "\n"

print("=" * 60)
print("Interactive RAG with SQL Database")
print("=" * 60)
print("\nDatabase Schema:")
print(schema_description)
print("You can ask questions about hybrid and electric cars.")
print("Type 'quit' or 'exit' to end the session.\n")

# System message for SQL generation
SQL_GENERATION_SYSTEM_MESSAGE = f"""
You are a SQL expert assistant. Your task is to convert natural language questions into valid SQLite queries.

Database schema:
{schema_description}

Important rules:
1. Generate ONLY the SQL query, no explanations or markdown formatting
2. Use proper SQLite syntax
3. Always use SELECT statements (no INSERT, UPDATE, DELETE, DROP)
4. Use appropriate WHERE, ORDER BY, LIMIT clauses as needed
5. For "fastest" questions, use ORDER BY acceleration ASC (lower is faster)
6. For "slowest" questions, use ORDER BY acceleration DESC
7. For "most efficient" or "best mpg", use ORDER BY mpg DESC
8. For "cheapest", use ORDER BY msrp ASC
9. For "most expensive", use ORDER BY msrp DESC
10. If asking for a single result or "the", add LIMIT 1
11. Use LIKE for partial matches (e.g., WHERE vehicle LIKE '%Prius%')
12. Consider conversation context to refine queries

Examples:
Q: "how fast is the prius v?"
A: SELECT vehicle, acceleration, year, mpg FROM cars WHERE vehicle LIKE '%Prius V%'

Q: "what is the fastest car?"
A: SELECT vehicle, acceleration, year, msrp FROM cars ORDER BY acceleration ASC LIMIT 1

Q: "show me efficient cars"
A: SELECT vehicle, mpg, year, class FROM cars WHERE mpg > 40 ORDER BY mpg DESC

Q: "give me the cheapest pickup truck"
A: SELECT vehicle, msrp, year, mpg FROM cars WHERE class = 'Pickup Truck' ORDER BY msrp ASC LIMIT 1
"""

ANSWER_SYSTEM_MESSAGE = """
You are a helpful assistant that answers questions about cars based on a hybrid car database.
You must use the data from the query results to answer the questions accurately.
Be concise and direct in your answers. Use the conversation history to provide context-aware responses.
"""

# Conversation history
messages = [{"role": "system", "content": ANSWER_SYSTEM_MESSAGE}]

while True:
    question = input("\nYour question about cars: ")

    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not question.strip():
        continue

    try:
        # Generate SQL query using the LLM with conversation context
        sql_response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SQL_GENERATION_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": f"New question: {question}\n\nConversation context: {messages[-4:] if len(messages) > 4 else messages}",
                },
            ],
        )

        sql_query = sql_response.choices[0].message.content.strip()
        # Remove any markdown code blocks if present
        if sql_query.startswith("```"):
            sql_query = sql_query.split("\n", 1)[1]
            sql_query = sql_query.rsplit("```", 1)[0]
        sql_query = sql_query.strip()

        print(f"\n🔍 Generated SQL: {sql_query}")

        # Execute the query
        cursor.execute(sql_query)
        results = cursor.fetchall()

        # Get column names
        column_names = [description[0] for description in cursor.description]

        # Format results as markdown table
        if results:
            results_table = " | ".join(column_names) + "\n"
            results_table += " | ".join("---" for _ in column_names) + "\n"
            for row in results[:10]:  # Limit to 10 rows for display
                results_table += " | ".join(str(val) if val is not None else "N/A" for val in row) + "\n"

            if len(results) > 10:
                results_table += f"\n... and {len(results) - 10} more rows"

            print(f"\n📊 Query results ({len(results)} rows):")
            print(results_table)
        else:
            results_table = "No results found."
            print(f"\n{results_table}")

        # Use the results to generate a natural language response
        messages.append({"role": "user", "content": f"{question}\n\nQuery results:\n{results_table}"})

        response = client.chat.completions.create(
            model=MODEL_NAME, temperature=0.3, messages=messages
        )

        llm_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": llm_response})

        print(f"\n💬 Response:")
        print(llm_response)

    except sqlite3.Error as e:
        print(f"\n❌ SQL Error: {e}")
        print("The generated query might be invalid. Please try rephrasing your question.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("An error occurred. Please try again.")

conn.close()
