import nltk
import typer
from pathlib import Path
import logging
import os
import shutil
from typing import List, Optional, Tuple
from dotenv import load_dotenv

from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from raglight.rag.rag import RAG
from typing_extensions import Annotated

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt as RichPrompt

from quo.prompt import Prompt
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline
from raglight.models.data_source_model import GitHubSource
from raglight.scrapper.github_scrapper import GithubScrapper


def download_nltk_resources_if_needed():
    """Download necessary NLTK resources if they are not already available."""
    required_resources = ["punkt", "stopwords"]
    for resource in required_resources:
        try:
            nltk.data.find(
                f"tokenizers/{resource}"
                if resource == "punkt"
                else f"corpora/{resource}"
            )
        except LookupError:
            console.print(
                f"[bold yellow]NLTK resource '{resource}' not found. Downloading...[/bold yellow]"
            )
            nltk.download(resource, quiet=True)
            console.print(
                f"[bold green]✅ Resource '{resource}' downloaded.[/bold green]"
            )


console = Console()


def simple_select(
    message: str, choices: List[str], default: Optional[str] = None
) -> str:
    """Prompt the user to select from a list using arrow keys."""
    from InquirerPy import inquirer

    return inquirer.select(message=message, choices=choices, default=default).execute()


def prompt_input():
    session = Prompt()
    return session.prompt(
        ">>> ", placeholder="<gray> enter your input here, type bye to quit</gray>"
    )


def print_llm_response(response: str):
    """Affiche la réponse LLM dans un panneau markdown cyan avec 🤖"""
    console.print(
        Panel(
            Markdown(response), border_style="cyan", title="[bold cyan]🤖[/bold cyan]"
        )
    )


def select_with_arrows(message, choices, default=None):
    """Prompt the user to select from a list."""
    return simple_select(message, choices, default)


def prompt_local_source() -> Path:
    cwd = os.getcwd()
    data_path_str = typer.prompt(
        "Enter the path to the directory with your documents", default=cwd
    )
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        console.print(
            f"[bold red]❌ Error: The path '{data_path_str}' is not a valid directory.[/bold red]"
        )
        raise typer.Exit(code=1)
    return data_path


def prompt_github_sources() -> List[GitHubSource]:
    github_sources: List[GitHubSource] = []
    console.print(
        "[cyan]Enter GitHub repository URLs (one per line, press Enter twice to finish):[/cyan]"
    )
    while True:
        repo_url = input("GitHub repo URL (or Enter to finish): ").strip()
        if not repo_url:
            break
        branch = typer.prompt(
            "Which branch should be used for this repository?", default="main"
        )
        github_sources.append(GitHubSource(url=repo_url, branch=branch))
    if github_sources:
        console.print(
            f"[green]✅ Added {len(github_sources)} GitHub repository(ies).[/green]"
        )
    return github_sources


def prompt_data_sources() -> Tuple[Optional[Path], List[GitHubSource]]:
    console.print("[bold cyan]\n--- 📂 Step 1: Data Source ---[/bold cyan]")
    source_type = simple_select(
        "Which knowledge source do you want to use?",
        choices=["Local folder", "GitHub repositories"],
        default="Local folder",
    )

    if source_type == "Local folder":
        return prompt_local_source(), []

    github_sources = prompt_github_sources()
    if not github_sources:
        console.print(
            "[bold red]❌ Error: At least one GitHub repository is required.[/bold red]"
        )
        raise typer.Exit(code=1)
    return None, github_sources


def ingest_github_sources(
    vector_store, github_sources: List[GitHubSource], ignore_folders: List[str]
) -> None:
    if not github_sources:
        return
    console.print("[bold cyan]⏳ Cloning GitHub repositories...[/bold cyan]")
    github_scrapper = GithubScrapper()
    github_scrapper.set_repositories(github_sources)
    repos_path = github_scrapper.clone_all()
    try:
        vector_store.ingest(data_path=repos_path, ignore_folders=ignore_folders)
        console.print("[bold green]✅ GitHub repositories indexed.[/bold green]")
    finally:
        shutil.rmtree(repos_path, ignore_errors=True)


app = typer.Typer(
    help="RAGLight CLI: An interactive wizard to index and chat with your documents."
)


def _safe_default(choices: list, value: str) -> str:
    """Return value if it is in choices, otherwise the first choice."""
    return value if value in choices else choices[0]


RAGLIGHT_BANNER = """[bold yellow]
██████╗  █████╗  ██████╗ ██╗     ██╗ ██████╗ ██╗  ██╗████████╗
██╔══██╗██╔══██╗██╔════╝ ██║     ██║██╔════╝ ██║  ██║╚══██╔══╝
██████╔╝███████║██║  ███╗██║     ██║██║  ███╗███████║   ██║
██╔══██╗██╔══██║██║   ██║██║     ██║██║   ██║██╔══██║   ██║
██║  ██║██║  ██║╚██████╔╝███████╗██║╚██████╔╝██║  ██║   ██║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
[/bold yellow][dim]  ⚡ Retrieval-Augmented Generation — fast, modular, local[/dim]
"""


@app.callback()
def callback():
    """
    RAGLight CLI application.
    """
    Settings.setup_logging()
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    _langfuse_configured = all(
        [
            os.environ.get("LANGFUSE_PUBLIC_KEY"),
            os.environ.get("LANGFUSE_SECRET_KEY"),
            os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL"),
        ]
    )
    if not _langfuse_configured:
        os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
    for name in [
        "telemetry",
        "langchain",
        "langchain_core",
        "langchain_core.tracing",
        "httpx",
        "urllib3",
        "requests",
        "chromadb",
        "chromadb.telemetry",
        "chromadb.telemetry.product.posthog",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)


@app.command(name="chat")
def interactive_chat_command():
    """
    Starts a guided, interactive session to configure, index, and chat with your data.
    Tip: set RAGLIGHT_* env vars (or a .env file) to skip the wizard entirely.
    """
    console.print(RAGLIGHT_BANNER)

    from raglight.api.server_config import ServerConfig

    load_dotenv(dotenv_path=Path(".env"))
    cfg = ServerConfig()

    _raglight_env_vars = [
        "RAGLIGHT_LLM_MODEL",
        "RAGLIGHT_LLM_PROVIDER",
        "RAGLIGHT_EMBEDDINGS_MODEL",
        "RAGLIGHT_EMBEDDINGS_PROVIDER",
        "RAGLIGHT_PERSIST_DIR",
        "RAGLIGHT_COLLECTION",
    ]
    headless = any(os.environ.get(v) for v in _raglight_env_vars)

    emb_choices = [
        Settings.HUGGINGFACE,
        Settings.OLLAMA,
        Settings.OPENAI,
        Settings.GOOGLE_GEMINI,
    ]
    llm_choices = [
        Settings.OLLAMA,
        Settings.MISTRAL,
        Settings.OPENAI,
        Settings.LMSTUDIO,
        Settings.GOOGLE_GEMINI,
    ]
    k_choices = ["5", "10", "15"]

    db_choices = [Settings.CHROMA, Settings.QDRANT]

    # ── Step 1: Model / DB config ──────────────────────────────────────────────
    if headless:
        db_type = cfg.db
        db_path = cfg.persist_dir
        db_host = cfg.db_host
        db_port = cfg.db_port
        collection = cfg.collection
        emb_provider = cfg.embeddings_provider
        embeddings_base_url = cfg.embeddings_api_base
        emb_model = cfg.embeddings_model
        llm_provider = cfg.llm_provider
        llm_base_url = cfg.llm_api_base
        llm_model = cfg.llm_model
        k = cfg.k

        console.print("[bold magenta]🚀 RAGLight Chat[/bold magenta]")
        console.print(
            f"  LLM          : [cyan]{llm_provider}[/cyan] / [cyan]{llm_model}[/cyan]"
        )
        console.print(
            f"  Embeddings   : [cyan]{emb_provider}[/cyan] / [cyan]{emb_model}[/cyan]"
        )
        console.print(f"  Vector store : [cyan]{db_type}[/cyan]")
        if db_host:
            console.print(f"  DB host      : [cyan]{db_host}:{db_port}[/cyan]")
        else:
            console.print(f"  Persist dir  : [cyan]{db_path}[/cyan]")
        console.print(f"  Collection   : [cyan]{collection}[/cyan]")
        console.print(f"  k            : [cyan]{k}[/cyan]")
    else:
        console.print(
            "[bold magenta]👋 Welcome to the RAGLight Interactive Setup Wizard![/bold magenta]"
        )
        console.print(
            "[magenta]I will guide you through setting up your RAG pipeline.[/magenta]"
        )
        console.print(
            "[dim]Tip: create a .env file with RAGLIGHT_* vars to skip the model setup.[/dim]\n"
        )

        console.print("[bold cyan]\n--- 💾 Step 1: Vector Database ---[/bold cyan]")
        db_type = simple_select(
            "Which vector store do you want to use?",
            choices=db_choices,
            default=_safe_default(db_choices, cfg.db),
        )
        db_path = typer.prompt(
            "Where should the vector database be stored?",
            default=cfg.persist_dir,
        )
        db_host = None
        db_port = None
        collection = typer.prompt(
            "What is the name for the database collection?",
            default=cfg.collection,
        )

        console.print("[bold blue]\n--- 🧠 Step 2: Embeddings Model ---[/bold blue]")
        emb_provider = simple_select(
            "Which embeddings provider do you want to use?",
            choices=emb_choices,
            default=_safe_default(emb_choices, cfg.embeddings_provider),
        )
        embeddings_base_url = RichPrompt.ask(
            "[bold]What is your base URL for the embeddings provider? (Not needed for HuggingFace)[/bold]",
            default=cfg.embeddings_api_base,
        )
        emb_model = RichPrompt.ask(
            "[bold]Which embedding model do you want to use?[/bold]",
            default=cfg.embeddings_model,
        )

        console.print(
            "[bold blue]\n--- 🤖 Step 3: Language Model (LLM) ---[/bold blue]"
        )
        llm_provider = simple_select(
            "Which LLM provider do you want to use?",
            choices=llm_choices,
            default=_safe_default(llm_choices, cfg.llm_provider),
        )
        llm_base_url = RichPrompt.ask(
            "[bold]What is your base URL for the LLM provider? (Not needed for Mistral)[/bold]",
            default=cfg.llm_api_base,
        )
        llm_model = RichPrompt.ask(
            "[bold]Which LLM do you want to use?[/bold]",
            default=cfg.llm_model,
        )
        k = int(
            simple_select(
                "How many documents should be retrieved for context (k)?",
                choices=k_choices,
                default=_safe_default(k_choices, str(cfg.k)),
            )
        )

    # ── Step 2: Knowledge base ─────────────────────────────────────────────────
    env_data_path = os.environ.get("RAGLIGHT_DATA_PATH")
    if env_data_path:
        data_path = Path(env_data_path)
        if not data_path.is_dir():
            console.print(
                f"[bold red]❌ RAGLIGHT_DATA_PATH '{env_data_path}' is not a valid directory.[/bold red]"
            )
            raise typer.Exit(code=1)
        github_sources = []
        ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS.copy()
        if headless:
            console.print(f"  Data path    : [cyan]{data_path}[/cyan]")
    else:
        data_path, github_sources = prompt_data_sources()
        ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS.copy()

    try:
        console.print("[bold cyan]\n--- ⏳ Indexing Documents ---[/bold cyan]")
        should_index = True
        if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
            console.print(f"[yellow]A database seems to exist at '{db_path}'.[/yellow]")
            if not typer.confirm(
                "Do you want to re-index the data? (This will add documents to the existing collection)\nIf you don't want, existing database will be used.",
                default=False,
            ):
                should_index = False

        vs_kwargs = {"collection_name": collection}
        if db_host:
            vs_kwargs["host"] = db_host
            if db_port:
                vs_kwargs["port"] = db_port
        else:
            vs_kwargs["persist_directory"] = db_path

        builder = Builder()
        builder.with_embeddings(
            emb_provider, model_name=emb_model, api_base=embeddings_base_url
        )
        builder.with_vector_store(db_type, **vs_kwargs)

        if should_index:
            vector_store = builder.build_vector_store()
            if data_path:
                vector_store.ingest(
                    data_path=str(data_path), ignore_folders=ignore_folders
                )
            ingest_github_sources(vector_store, github_sources, ignore_folders)
            console.print("[bold green]✅ Indexing complete.[/bold green]")
        else:
            console.print(
                "[bold yellow]Skipping indexing, using existing database.[/bold yellow]"
            )

        console.print(
            "[bold cyan]\n--- 💬 Step 6: Starting Chat Session ---[/bold cyan]"
        )

        rag_pipeline: RAG = builder.with_llm(
            llm_provider,
            model_name=llm_model,
            api_base=llm_base_url,
            system_prompt=Settings.DEFAULT_SYSTEM_PROMPT,
        ).build_rag(k=k)

        console.print(
            "[bold green]✅ RAG pipeline is ready. You can start chatting now![/bold green]"
        )
        console.print("[yellow]Type 'quit' or 'exit' to end the session.\n[/yellow]")

        while True:
            query = prompt_input()
            if query.lower() in ["bye", "exit", "quit"]:
                console.print("🤖 : See you soon 👋")
                break

            full_response = ""
            with Live(
                Panel(
                    Markdown(""),
                    border_style="cyan",
                    title="[bold cyan]🤖[/bold cyan]",
                ),
                console=console,
                refresh_per_second=15,
                vertical_overflow="visible",
            ) as live:
                for chunk in rag_pipeline.generate_streaming(query):
                    full_response += chunk
                    live.update(
                        Panel(
                            Markdown(full_response),
                            border_style="cyan",
                            title="[bold cyan]🤖[/bold cyan]",
                        )
                    )

    except Exception as e:
        console.print(f"[bold red]❌ An unexpected error occurred: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(name="agentic-chat")
def interactive_agentic_chat_command():
    """
    Starts a guided, interactive session to configure, index, and chat with your data.
    Tip: set RAGLIGHT_* env vars (or a .env file) to skip the wizard entirely.
    """
    console.print(RAGLIGHT_BANNER)

    from raglight.api.server_config import ServerConfig

    load_dotenv(dotenv_path=Path(".env"))
    cfg = ServerConfig()

    _raglight_env_vars = [
        "RAGLIGHT_LLM_MODEL",
        "RAGLIGHT_LLM_PROVIDER",
        "RAGLIGHT_EMBEDDINGS_MODEL",
        "RAGLIGHT_EMBEDDINGS_PROVIDER",
        "RAGLIGHT_PERSIST_DIR",
        "RAGLIGHT_COLLECTION",
        "RAGLIGHT_MCP_URLS",
    ]
    headless = any(os.environ.get(v) for v in _raglight_env_vars)

    emb_choices = [
        Settings.HUGGINGFACE,
        Settings.OLLAMA,
        Settings.OPENAI,
        Settings.GOOGLE_GEMINI,
    ]
    llm_choices = [
        Settings.OLLAMA,
        Settings.MISTRAL,
        Settings.OPENAI,
        Settings.LMSTUDIO,
        Settings.GOOGLE_GEMINI,
    ]
    k_choices = ["5", "10", "15"]

    db_choices = [Settings.CHROMA, Settings.QDRANT]

    # ── Step 1: Model / DB config ──────────────────────────────────────────────
    if headless:
        db_type = cfg.db
        db_path = cfg.persist_dir
        db_host = cfg.db_host
        db_port = cfg.db_port
        collection = cfg.collection
        emb_provider = cfg.embeddings_provider
        embeddings_base_url = cfg.embeddings_api_base
        emb_model = cfg.embeddings_model
        llm_provider = cfg.llm_provider
        llm_base_url = cfg.llm_api_base
        llm_model = cfg.llm_model
        k = cfg.k
        mcp_urls_env = cfg.mcp_urls
        mcp_config = {}
        if mcp_urls_env:
            for idx, url in enumerate(mcp_urls_env.split(",")):
                url = url.strip()
                if url:
                    mcp_config[f"mcp_server_{idx}"] = {"url": url, "transport": "sse"}

        console.print("[bold magenta]🚀 RAGLight Agentic Chat[/bold magenta]")
        console.print(
            f"  LLM          : [cyan]{llm_provider}[/cyan] / [cyan]{llm_model}[/cyan]"
        )
        console.print(
            f"  Embeddings   : [cyan]{emb_provider}[/cyan] / [cyan]{emb_model}[/cyan]"
        )
        console.print(f"  Vector store : [cyan]{db_type}[/cyan]")
        if db_host:
            console.print(f"  DB host      : [cyan]{db_host}:{db_port}[/cyan]")
        else:
            console.print(f"  Persist dir  : [cyan]{db_path}[/cyan]")
        console.print(f"  Collection   : [cyan]{collection}[/cyan]")
        console.print(f"  k            : [cyan]{k}[/cyan]")
    else:
        console.print(
            "[bold magenta]👋 Welcome to the RAGLight Interactive Setup Wizard![/bold magenta]"
        )
        console.print(
            "[magenta]I will guide you through setting up your RAG pipeline.[/magenta]"
        )
        console.print(
            "[dim]Tip: create a .env file with RAGLIGHT_* vars to skip the model setup.[/dim]\n"
        )

        console.print("[bold cyan]\n--- 💾 Step 1: Vector Database ---[/bold cyan]")
        db_type = simple_select(
            "Which vector store do you want to use?",
            choices=db_choices,
            default=_safe_default(db_choices, cfg.db),
        )
        db_path = typer.prompt(
            "Where should the vector database be stored?",
            default=cfg.persist_dir,
        )
        db_host = None
        db_port = None
        collection = typer.prompt(
            "What is the name for the database collection?",
            default=cfg.collection,
        )

        console.print("[bold blue]\n--- 🧠 Step 2: Embeddings Model ---[/bold blue]")
        emb_provider = simple_select(
            "Which embeddings provider do you want to use?",
            choices=emb_choices,
            default=_safe_default(emb_choices, cfg.embeddings_provider),
        )
        embeddings_base_url = RichPrompt.ask(
            "[bold]What is your base URL for the embeddings provider? (Not needed for HuggingFace)[/bold]",
            default=cfg.embeddings_api_base,
        )
        emb_model = RichPrompt.ask(
            "[bold]Which embedding model do you want to use?[/bold]",
            default=cfg.embeddings_model,
        )

        console.print(
            "[bold blue]\n--- 🤖 Step 3: Language Model (LLM) ---[/bold blue]"
        )
        llm_provider = simple_select(
            "Which LLM provider do you want to use?",
            choices=llm_choices,
            default=_safe_default(llm_choices, cfg.llm_provider),
        )
        llm_base_url = RichPrompt.ask(
            "[bold]What is your base URL for the LLM provider? (Not needed for Mistral)[/bold]",
            default=cfg.llm_api_base,
        )
        llm_model = RichPrompt.ask(
            "[bold]Which LLM do you want to use?[/bold]",
            default=cfg.llm_model,
        )
        k = int(
            simple_select(
                "How many documents should be retrieved for context (k)?",
                choices=k_choices,
                default=_safe_default(k_choices, str(cfg.k)),
            )
        )

        console.print("[bold blue]\n--- 🔌 Step 4: MCP Servers ---[/bold blue]")
        mcp_config = {}
        if typer.confirm("Do you want to configure an MCP server?", default=False):
            mcp_url = typer.prompt(
                "What is the URL of your MCP server? (e.g. http://127.0.0.1:8001/sse)"
            )
            if mcp_url:
                mcp_config["mcp_server_0"] = {"url": mcp_url, "transport": "sse"}
            idx = 1
            while typer.confirm(
                "Do you want to add another MCP server?", default=False
            ):
                mcp_url = typer.prompt("What is the URL of your MCP server?")
                if mcp_url:
                    mcp_config[f"mcp_server_{idx}"] = {
                        "url": mcp_url,
                        "transport": "sse",
                    }
                    idx += 1

    # ── Step 2: Knowledge base ─────────────────────────────────────────────────
    env_data_path = os.environ.get("RAGLIGHT_DATA_PATH")
    if env_data_path:
        data_path = Path(env_data_path)
        if not data_path.is_dir():
            console.print(
                f"[bold red]❌ RAGLIGHT_DATA_PATH '{env_data_path}' is not a valid directory.[/bold red]"
            )
            raise typer.Exit(code=1)
        github_sources = []
        ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS.copy()
        if headless:
            console.print(f"  Data path    : [cyan]{data_path}[/cyan]")
    else:
        data_path, github_sources = prompt_data_sources()
        ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS.copy()

    # Derive API key from provider (reads from env vars via Settings)
    api_key = None
    if llm_provider == Settings.OPENAI:
        api_key = Settings.OPENAI_API_KEY
    elif llm_provider == Settings.GOOGLE_GEMINI:
        api_key = Settings.GEMINI_API_KEY

    try:
        console.print("[bold cyan]\n--- ⏳ Indexing Documents ---[/bold cyan]")
        should_index = True
        if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
            console.print(f"[yellow]A database seems to exist at '{db_path}'.[/yellow]")
            if not typer.confirm(
                "Do you want to re-index the data? (This will add documents to the existing collection)\nIf you don't want, existing database will be used.",
                default=False,
            ):
                should_index = False

        vector_store_config = VectorStoreConfig(
            embedding_model=emb_model,
            api_base=embeddings_base_url,
            database=db_type,
            persist_directory=db_path if not db_host else None,
            host=db_host,
            port=db_port,
            provider=emb_provider,
            collection_name=collection,
        )

        config = AgenticRAGConfig(
            provider=llm_provider,
            model=llm_model,
            k=k,
            system_prompt=Settings.DEFAULT_AGENT_PROMPT,
            max_steps=4,
            api_key=api_key,
            api_base=llm_base_url,
            mcp_config=mcp_config if mcp_config else None,
        )

        agenticRag = AgenticRAGPipeline(config, vector_store_config)

        if should_index:
            if data_path:
                agenticRag.get_vector_store().ingest(
                    data_path=str(data_path), ignore_folders=ignore_folders
                )
            ingest_github_sources(
                agenticRag.get_vector_store(), github_sources, ignore_folders
            )
            console.print("[bold green]✅ Indexing complete.[/bold green]")
        else:
            console.print(
                "[bold yellow]Skipping indexing, using existing database.[/bold yellow]"
            )

        console.print(
            "[bold cyan]\n--- 💬 Step 6: Starting Chat Session ---[/bold cyan]"
        )

        console.print(
            "[bold green]✅ RAG pipeline is ready. You can start chatting now![/bold green]"
        )
        console.print("[yellow]Type 'quit' or 'exit' to end the session.\n[/yellow]")

        while True:
            query = prompt_input()
            if query.lower() in ["bye", "exit", "quit"]:
                console.print("🤖 : See you soon 👋")
                break

            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold cyan]Waiting for response...[/bold cyan]"),
                transient=True,
                console=console,
            ) as progress:
                task = progress.add_task("", total=None)
                response = agenticRag.generate(query)
                progress.update(task, completed=1)

            print_llm_response(response)

    except Exception as e:
        console.print(f"[bold red]❌ An unexpected error occurred: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command(name="serve")
def serve_command(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", help="Port to listen on"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload (dev mode)"
    ),
    workers: int = typer.Option(1, "--workers", help="Number of worker processes"),
    ui: bool = typer.Option(False, "--ui", help="Start Streamlit UI alongside the API"),
    ui_port: int = typer.Option(8501, "--ui-port", help="Port for the Streamlit UI"),
):
    """
    Start the RAGLight REST API server (configured via RAGLIGHT_* env vars).
    Use --ui to also launch the Streamlit chat interface.
    Langfuse tracing is enabled automatically when LANGFUSE_HOST (or LANGFUSE_BASE_URL),
    LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in the environment.
    """
    import signal
    import subprocess
    import sys
    import uvicorn
    from raglight.api.server_config import ServerConfig

    load_dotenv(dotenv_path=Path(".env"))
    config = ServerConfig()
    display_host = "localhost" if host == "0.0.0.0" else host

    console.print("[bold magenta]🚀 RAGLight API Server[/bold magenta]")
    console.print(
        f"  LLM          : [cyan]{config.llm_provider}[/cyan] / [cyan]{config.llm_model}[/cyan]"
    )
    console.print(
        f"  Embeddings   : [cyan]{config.embeddings_provider}[/cyan] / [cyan]{config.embeddings_model}[/cyan]"
    )
    console.print(f"  Vector store : [cyan]{config.db}[/cyan]")
    if config.db_host:
        console.print(
            f"  DB host      : [cyan]{config.db_host}:{config.db_port}[/cyan]"
        )
    else:
        console.print(f"  Persist dir  : [cyan]{config.persist_dir}[/cyan]")
    console.print(f"  Collection   : [cyan]{config.collection}[/cyan]")
    console.print(f"  k            : [cyan]{config.k}[/cyan]")
    if config.langfuse_host:
        if config.langfuse_public_key and config.langfuse_secret_key:
            console.print(f"  Langfuse     : [cyan]{config.langfuse_host}[/cyan]")
        else:
            console.print(
                "  Langfuse     : [bold yellow]host set but LANGFUSE_PUBLIC_KEY / "
                "LANGFUSE_SECRET_KEY are missing — tracing disabled[/bold yellow]"
            )

    if not ui:
        console.print(f"\n[bold green]Listening on http://{host}:{port}[/bold green]\n")
        uvicorn.run(
            "raglight.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
        )
        return

    console.print(f"\n  API  →  [bold green]http://{display_host}:{port}[/bold green]")
    console.print(f"  UI   →  [bold cyan]http://{display_host}:{ui_port}[/bold cyan]\n")

    env = {**os.environ, "RAGLIGHT_API_URL": f"http://localhost:{port}"}

    streamlit_app = str(Path(__file__).parent.parent / "ui" / "streamlit_app.py")

    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "raglight.api.app:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        uvicorn_cmd.append("--reload")
    elif workers > 1:
        uvicorn_cmd += ["--workers", str(workers)]

    api_proc = subprocess.Popen(uvicorn_cmd, env=env)
    ui_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            streamlit_app,
            "--server.port",
            str(ui_port),
            "--server.headless",
            "true",
        ],
        env=env,
    )

    def _shutdown(sig, frame):
        api_proc.terminate()
        ui_proc.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    api_proc.wait()
    ui_proc.wait()


if __name__ == "__main__":
    app()
