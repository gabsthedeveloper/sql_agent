import duckdb
import getpass
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from rich.console import Console

load_dotenv()
console = Console()

LLM_URL = os.getenv('LLM_URL')
LLM_MODEL = os.getenv('LLM_MODEL')
LLM_API_KEY = os.getenv('LLM_API_KEY')
CSV_FILE_PATH = "data.csv"


# Define the model
model = OllamaModel(
    model_name=LLM_MODEL,
    provider=OllamaProvider(
        base_url=LLM_URL,
        api_key=LLM_API_KEY
    )
)


# SCHEMAS & DEPENDENCIES
class RunTelemetry(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_prompt: str
    generated_sql: str
    sql_success: bool
    db_error: str | None = None
    user_rating: int | None = None  # 1 for Good, -1 for Bad
    user_comment: str | None = None


class SQLGenerationResult(BaseModel):
    explanation: str = Field(description="Brief explanation of the calculation strategy.")
    sql_query: str = Field(description="The final verified, working DuckDB SQL query string.")


@dataclass
class AgentDependencies:
    db_connection: duckdb.DuckDBPyConnection
    table_name: str
    schema_info: str


# SELF-CORRECTING SQL AGENT
sql_generation_agent = Agent(
    model=model,
    deps_type=AgentDependencies,
    output_type=SQLGenerationResult,
    system_prompt=(
        "You are an expert data analyst. Your job is to generate a valid DuckDB SQL query.\n"
        "Never exclude NULL values in queries\n"
        "Crucial: You must use the `test_sql_query` tool to execute and verify your query works before finalizing your answer.\n"
        "Also crucial: if the query output is 0 and there is no error, please use that query.\n"
        "If `test_sql_query` returns a database error, read the error message, rewrite your SQL query, and test it again."
    )
)


# SELF-CORRECTION TOOL
@sql_generation_agent.tool
def test_sql_query(ctx: RunContext[AgentDependencies], query: str) -> str:
    """
    Executes a SQL query against the database to verify if it is valid.
    Use this tool to test your query before providing your final structured answer.
    """
    try:
        # Normalize and convert to lowercase for inspection
        lower_query = query.strip().lower()

        # 1. Enforce read-only queries (disallow DROP, DELETE, INSERT, UPDATE, etc.)
        if not re.match(r'^(select|with)\b', lower_query):
            return "DATABASE ERROR: Only read-only queries (SELECT or WITH) are allowed."

        # 2. FAIL-SAFE: Programmatically block reading from files directly
        if re.search(r'\b(read_csv|read_json|read_csv_auto|read_json_auto)\b|\.(csv|json)\b', lower_query):
            return (
                "DATABASE ERROR: You are attempting to read raw files directly."
		"This is strictly forbidden. You must SELECT only from the database"
            )

        # Execute query if checks pass
        res = ctx.deps.db_connection.execute(query).fetchdf()
        return f"SUCCESS! Query execution works. Sample Output:\n{res.head(2).to_string(index=False)}"

    except Exception as e:
        return f"DATABASE ERROR: {str(e)}. Please correct your SQL syntax or column names and try again."


# RE-RUN PIPELINE
@sql_generation_agent.system_prompt
def inject_dynamic_schema(ctx: RunContext[AgentDependencies]) -> str:
    return (
        f"CRITICAL RULES FOR SQL GENERATION:\n"
        f"1. You MUST query the database table named '{ctx.deps.table_name}' directly.\n"
        f"2. DO NOT use functions like `read_csv_auto()`, `read_json_auto()`, or reference '{CSV_FILE_PATH}'.\n\n"
        
        f"Valid Columns in '{ctx.deps.table_name}':\n{ctx.deps.schema_info}"
    )


if __name__ == "__main__":
    # Get data
    file_path = Path(CSV_FILE_PATH)
    if not file_path.is_file():
        sys.exit("Error: Critical dependency file '{CSV_FILE_PATH}' is missing.")

    # Connect to database
    db_connect = duckdb.connect(database=':memory:')

    # Introspect schema safely
    db_connect.execute(f"CREATE TABLE nodes AS SELECT * FROM read_csv_auto('{CSV_FILE_PATH}')")
    raw_schema = db_connect.execute(f"DESCRIBE nodes").fetchall()
    schema_text = "\n".join([f"- Column '{col[0]}' ({col[1]})" for col in raw_schema])
    dependencies = AgentDependencies(db_connection=db_connect, table_name="nodes", schema_info=schema_text)

    # Initialize chat history
    message_history = []
    # SUMMARY AGENT
    summary_agent = Agent(
        model=model,
        system_prompt="Summarize this structured analytical data clearly for the user."
    )

    # Initialize conversation with a personalized greeting message
    username = getpass.getuser()
    greeting = f"Hello {username}, how can I help you?"
    console.print(f"[blue]Assistant:[/blue] {greeting}")

    while True:
        console.print(f"[green]User:[/green] ", end="")
        user_input = console.input()
        # Continue if user input contains nothing
        if not user_input.strip():
            continue
        # Exit agent if user inputs 'exit' or 'quit'
        if user_input.lower() in ['quit', 'exit']:
            console.print("[dim]Goodbye![/dim]")
            break
        
        # Run the agent with existing history
        with console.status("[dim]Thinking...[/dim]", spinner="arc"):
            generation_result = sql_generation_agent.run_sync(user_input, deps=dependencies, message_history=message_history)
            verified_sql = generation_result.output.sql_query
            #print(f"--- Final Verified SQL Query ---\n{verified_sql}\n")
            console.print(f"\n[cyan]--- Final Verified SQL Query ---[/cyan]\n{verified_sql}\n")

            # Run the finalized, working query
            sql_output = db_connect.execute(verified_sql).fetchdf()
            
            # Summarize the working data
            user_prompt = f"Question: {user_input}\nData:\n{sql_output.to_string(index=False)}"
            final_output = summary_agent.run_sync(user_prompt)
        
        # Update history with the new turn
        message_history = generation_result.all_messages()
        console.print(f"[blue]Assistant:[/blue] {final_output.output}")
