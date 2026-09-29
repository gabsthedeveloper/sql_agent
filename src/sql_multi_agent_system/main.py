import duckdb
import getpass
import logfire
import os
import sqlglot
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
from sqlglot import exp
from sqlglot.errors import ParseError

load_dotenv()
logfire.configure(console=False)
logfire.instrument_pydantic_ai()

LLM_URL = os.getenv('LLM_URL')
LLM_MODEL = os.getenv('LLM_MODEL')
LLM_API_KEY = os.getenv('LLM_API_KEY')
FORBIDDEN_FUNCTIONS = {"read_csv","read_csv_auto","read_json","read_json_auto","read_parquet","scan_parquet","scan_csv"}
CSV_FILE_PATH = "data.csv"


console = Console()
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


@logfire.instrument()
def validate_ast_sql(query: str, allowed_table: str = "nodes") -> tuple[bool, str]:
    """
    Validates a SQL query using AST parsing.
    Returns (is_valid, error_message).
    """
    # 1. Parse into AST using DuckDB dialect
    try:
        statements = [s for s in sqlglot.parse(query, read="duckdb") if s is not None]
    except ParseError as e:
        logfire.warn("SQL parsing failed: {error}", error=str(e)) # Optional: add a specific log
        return False, f"DATABASE ERROR: SQL syntax error. {str(e)}"

    # 2. Block empty or multi-statement injections
    if len(statements) == 0:
        return False, "DATABASE ERROR: Empty query supplied."
    if len(statements) > 1:
        return False, "DATABASE ERROR: Multi-statement execution is strictly forbidden."

    parsed = statements[0]

    # 3. Enforce strictly read-only root queries
    if not isinstance(parsed, (exp.Select, exp.Union)):
        return False, f"DATABASE ERROR: Only read-only SELECT or WITH statements allowed. Received: {parsed.key.upper()}."

    # 4. Check for forbidden file-reading functions
    for func in parsed.find_all(exp.Func, exp.Anonymous):
        func_name = func.name.lower() if func.name else ""
        if func_name in FORBIDDEN_FUNCTIONS:
            return False, f"DATABASE ERROR: Function '{func_name}' is forbidden. Direct file reads are not allowed."

    # 5. Block direct file paths in FROM/JOIN
    for literal in parsed.find_all(exp.Literal):
        if isinstance(literal.this, str):
            val = literal.this.lower()
            if any(val.endswith(ext) for ext in [".csv", ".json", ".parquet"]):
                return False, f"DATABASE ERROR: Query references file path '{literal.this}'. Query the '{allowed_table}' table instead."

    # 6. Extract CTE names
    cte_names = {cte.alias.lower() for cte in parsed.find_all(exp.CTE) if cte.alias}

    # 7. Extract all referenced tables
    referenced_tables = {table.name.lower() for table in parsed.find_all(exp.Table) if table.name}

    # Verify that the base table is actually queried
    if allowed_table.lower() not in referenced_tables:
        return False, f"DATABASE ERROR: Query does not target '{allowed_table}'. You must include 'FROM {allowed_table}'."

    # Verify no unauthorized external tables are being accessed
    unauthorized_tables = referenced_tables - cte_names - {allowed_table.lower()}
    if unauthorized_tables:
        return False, f"DATABASE ERROR: Unauthorized table(s): {', '.join(unauthorized_tables)}. Target '{allowed_table}' only."

    return True, ""


# SELF-CORRECTION TOOL
@sql_generation_agent.tool
def test_sql_query(ctx: RunContext[AgentDependencies], query: str) -> str:
    """
    Executes a SQL query against the database to verify if it is valid.
    Use this tool to test your query before providing your final structured answer.
    """
    try:
        # Run AST verification
        is_valid, error_msg = validate_ast_sql(query, allowed_table=ctx.deps.table_name)
        if not is_valid:
            logfire.warn("AST validation failed for query: {query}", query=query)
            return error_msg

        # If AST passes, execute against DuckDB
        with logfire.span("Executing verified query on DuckDB"):
            res = ctx.deps.db_connection.execute(query).fetchdf()
        return f"SUCCESS! Query execution works. Sample Output:\n{res.head(2).to_string(index=False)}"

    except Exception as e:
        logfire.error("Exception during SQL query execution: {error}", error=str(e))
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
        console.print("[bold red]Could not create or find the nodes data file. Exiting.[/bold red]")
        exit()

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
        with logfire.span("User Input", user_input=user_input):
            with console.status("[dim]Thinking...[/dim]", spinner="arc"):
                try:
                    generation_result = sql_generation_agent.run_sync(user_input, deps=dependencies, message_history=message_history)
                    verified_sql = generation_result.output.sql_query

                    # Run the finalized, working query
                    sql_output = db_connect.execute(verified_sql).fetchdf()
                    
                    # Summarize the working data
                    user_prompt = f"Question: {user_input}\nData:\n{sql_output.to_string(index=False)}"
                    final_output = summary_agent.run_sync(user_prompt)
                except Exception as e:
                    console.print(f"[red]Error during SQL generation pipeline: {e}[/red]")
                    continue
            
            # Output results
            console.print(f"\n[cyan]--- Final Verified SQL Query ---[/cyan]\n{verified_sql}\n")
            console.print(f"[blue]Assistant:[/blue] {final_output.output}")

            # Update history with the new input
            message_history = generation_result.all_messages()
