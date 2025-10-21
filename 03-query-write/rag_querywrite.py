import csv
import os

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

# Index the CSV data
with csv_path.open(newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    rows = list(reader)
documents = [{"id": (i + 1), "body": " ".join(row)} for i, row in enumerate(rows[1:])]
index = lunr(ref="id", fields=["body"], documents=documents)

def search(query):
    # Search the index for the user's question
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

while True:
    question = input("\nYour question about electric cars: ") # Example: "give me the cheapes pick uptruck"

    # Rewrite the query to fix typos and incorporate past context
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.05,
        messages=[
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": f"New user question:{question}\n\nConversation history:{messages}",
            },
        ],
    )
    search_query = response.choices[0].message.content
    print(f"Rewritten query: {search_query}")

    # Search the CSV for the question
    matches = search(search_query)
    print("Matches found:\n", matches)

    # Use the matches to generate a response
    messages.append({"role": "user", "content": f"{question}\nSources: {matches}"})
    response = client.chat.completions.create(model=MODEL_NAME, temperature=0.3, messages=messages)

    llm_response = response.choices[0].message.content
    messages.append({"role": "assistant", "content": llm_response})

    print(f"\nResponse from LLM:")
    print(llm_response)