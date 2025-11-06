# Exercise 3: Interactive RAG with SQL

This exercise demonstrates an interactive conversational interface using RAG with SQL query generation.

## What's Different from Exercise 2?

1. **Interactive Loop**: Continuous conversation with the system
2. **Context Awareness**: Maintains conversation history for follow-up questions
3. **Better UX**: Shows generated SQL and formatted results
4. **Error Handling**: Gracefully handles SQL errors

## Files

- `rag_sql_interactive.py` - Interactive RAG with SQL query generation
- Uses the same database from Exercise 2

## Setup

Same as Exercise 2. Make sure you have:
1. `.env` file with `GITHUB_TOKEN` in the root directory
2. Dependencies installed
3. Database created (run `create_database.py` in Exercise 2)

## Running the Example

```bash
python rag_sql_interactive.py
```

## Example Conversation

```
Your question about cars: what's the fastest car?
🔍 Generated SQL: SELECT vehicle, acceleration, year, msrp FROM cars ORDER BY acceleration ASC LIMIT 1
📊 Query results (1 rows):
vehicle | acceleration | year | msrp
--- | --- | --- | ---
Freed/Freed Spike | 6.29 | 2011 | 27972.07
💬 Response:
The fastest car in the database is the Freed/Freed Spike from 2011, with an acceleration time of 6.29 seconds (0-60 mph).

Your question about cars: what about the most efficient one?
🔍 Generated SQL: SELECT vehicle, mpg, year, msrp FROM cars ORDER BY mpg DESC LIMIT 1
📊 Query results (1 rows):
vehicle | mpg | year | msrp
--- | --- | --- | ---
Prius V | 72.92 | 2011 | 30588.35
💬 Response:
The most efficient car is the Prius V from 2011, with an impressive 72.92 miles per gallon.
```

## Features

- **Natural Language Queries**: Ask questions in plain English
- **SQL Generation**: Automatic conversion to SQL queries
- **Query Visibility**: See the generated SQL for learning
- **Conversation History**: Follow-up questions understand context
- **Results Preview**: See query results before the final answer
- **Error Handling**: Helpful error messages for invalid queries

## Tips for Asking Questions

- Be specific: "cheapest SUV" instead of "cheap car"
- Use comparisons: "cars with mpg over 40"
- Filter by attributes: "pickup trucks from 2010"
- Ask for rankings: "top 5 most expensive cars"
- Follow up: After asking about a car, you can ask "what year?" or "how much?"
