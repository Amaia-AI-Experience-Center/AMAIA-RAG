import csv
import os

import azure.identity
import openai
from dotenv import load_dotenv
from lunr import lunr
from pathlib import Path

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")
client = openai.OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4o")

# Path to the folder where this script lives
script_dir = Path(__file__).parent

# Path to the CSV inside the 'data' folder at the same level as the script
csv_path = script_dir / "data" / "hybrid.csv"

# Open the CSV
with open(csv_path) as file:
    reader = csv.reader(file)
    rows = list(reader)
headers = rows[0]
documents = [
    {
        "id": i + 1,
        "vehicle": row[0],
        "year": row[1],
        "msrp": row[2],
        "acceleration": row[3],
        "mpg": row[4],
        "class": row[5],
    }
    for i, row in enumerate(rows[1:])
]

index = lunr(
    ref="id",
    fields=["vehicle", "year", "msrp", "acceleration", "mpg", "class"],
    documents=documents,
)


def search(query):
    # Search the index for the user question
    results = index.search(query)
    matching_rows = [rows[int(result["ref"])] for result in results]

    # Format as a markdown table, since language models understand markdown
    matches_table = " | ".join(rows[0]) + "\n" + " | ".join(" --- " for _ in range(len(rows[0]))) + "\n"
    matches_table += "\n".join(" | ".join(row) for row in matching_rows)
    return matches_table


QUERY_REWRITE_SYSTEM_MESSAGE = """
You are a helpful assistant that rewrites user questions into concise keyword queries
to search over a CSV of hybrid and electric car data.
The CSV has these columns: vehicle, year, msrp, acceleration, mpg, class.
Only include column names or possible values, in lowercase, separated by spaces.
Do not include words like 'car', 'vehicle', 'electric', 'fastest', 'slowest', 'data', or punctuation.
Examples:
- "which is the fastest car" → "acceleration"
- "show me efficient cars" → "mpg"
- "cars from 2000" → "year 2000"
Respond with ONLY the keyword query.
"""

SYSTEM_MESSAGE = """
You are a helpful assistant that answers questions about cars based off a hybrid car data set.
You must use the data set to answer the questions, you should not provide any info that is not in the provided sources.
"""
messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

def handle_numeric_query(question, rows):
    """Handle questions that ask for max/min based on numeric fields."""
    question = question.lower()

    if "fast" in question or "acceleration" in question:
        # Find minimum acceleration (lower = faster)
        min_row = min(rows[1:], key=lambda r: float(r[3]))
        return f"The fastest car is {min_row[0]} ({min_row[1]}) with an acceleration of {min_row[3]} seconds."

    if "efficient" in question or "mpg" in question:
        # Find max MPG (higher = more efficient)
        max_row = max(rows[1:], key=lambda r: float(r[4]))
        return f"The most fuel-efficient car is {max_row[0]} ({max_row[1]}) with {max_row[4]} MPG."

    return None


while True:
    question = input("\nYour question about electric cars: ")

    numeric_answer = handle_numeric_query(question, rows)
    if numeric_answer:
        print(f"\nResponse from {API_HOST} {MODEL_NAME}: \n")
        print(numeric_answer)
    continue


    # Rewrite the query to fix typos and incorporate past context
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.05,
        messages=[
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_MESSAGE},
            {"role": "user", "content": f"New user question:{question}\n\nConversation history:{messages}"},
        ],
    )
    search_query = response.choices[0].message.content
    print(f"Rewritten query: {search_query}")

    # Search the CSV for the question
    matches = search(search_query)
    print("Found matches:\n", matches)

    # Use the matches to generate a response
    messages.append({"role": "user", "content": f"{question}\nSources: {matches}"})
    response = client.chat.completions.create(model=MODEL_NAME, temperature=0.3, messages=messages)

    bot_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": bot_response})

    print(f"\nResponse from {API_HOST} {MODEL_NAME}: \n")
    print(bot_response)
