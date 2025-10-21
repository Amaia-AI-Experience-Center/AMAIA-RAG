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
documents = [{"id": (i + 1), "body": " ".join(row)} for i, row in enumerate(rows[1:])]
index = lunr(ref="id", fields=["body"], documents=documents)

# Get the user question
user_question = "how fast is the prius v?"

# Search the index for the user question
results = index.search(user_question)
matching_rows = [rows[int(result["ref"])] for result in results]

# Format as a markdown table, since language models understand markdown
matches_table = " | ".join(rows[0]) + "\n" + " | ".join(" --- " for _ in range(len(rows[0]))) + "\n"
matches_table += "\n".join(" | ".join(row) for row in matching_rows)

print("Found matches:")
print(matches_table)

# Now we can use the matches to generate a response
SYSTEM_MESSAGE = """
You are a helpful assistant that answers questions about cars based off a hybrid car data set.
You must use the data set to answer the questions, you should not provide any info that is not in the provided sources.
"""

response = client.chat.completions.create(
    model=MODEL_NAME,
    temperature=0.3,
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{user_question}\nSources: {matches_table}"},
    ],
)

print(f"\nResponse from {API_HOST}: \n")
print(response.choices[0].message.content)
