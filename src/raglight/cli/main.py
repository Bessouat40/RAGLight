import nltk
import typer
from pathlib import Path
import logging
import os
import questionary

from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from raglight.rag.rag import RAG
from typing_extensions import Annotated

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from quo.prompt import Prompt
from raglight.config.agentic_rag_config import AgenticRAGConfig
from raglight.config.vector_store_config import VectorStoreConfig
from raglight.rag.simple_agentic_rag_api import AgenticRAGPipeline

# Static mapping of providers to popular embedding models
EMBEDDING_PROVIDERS = {
    "HuggingFace": [
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-base-v2",
        "thenlper/gte-base"
    ],
    "Ollama": [
        "llama3",
        "nomic-embed-text",
        "mxbai-embed-large"
    ],
    "OpenAI": [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ]
}

# Static mapping of LLM providers to popular models
LLM_PROVIDERS = {
    "Ollama": [
        "llama3",
        "mistral",
        "phi3",
        "gemma",
        "llama2",
        "codellama"
    ],
    "OpenAI": [
        "gpt-4o",
        "gpt-4",
        "gpt-3.5-turbo"
    ],
    "Mistral": [
        "mistral-tiny",
        "mistral-small",
        "mistral-medium"
    ]
}


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
    """Prompt the user to select from a list using arrow keys."""
    return questionary.select(
        message,
        choices=choices,
        default=default
    ).ask()


app = typer.Typer(
    help="RAGLight CLI: An interactive wizard to index and chat with your documents."
)


@app.callback()
def callback():
    """
    RAGLight CLI application.
    """
    Settings.setup_logging()
    for name in [
        "telemetry",
        "langchain",
        "langchain_core",
        "langchain_core.tracing",
        "httpx",
        "urllib3",
        "requests",
        "chromadb",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)


@app.command(name="chat")
def interactive_chat_command():
    """
    Starts a guided, interactive session to configure, index, and chat with your data.
    """
    console.print(
        "[bold magenta]👋 Welcome to the RAGLight Interactive Setup Wizard![/bold magenta]"
    )
    console.print(
        "[magenta]I will guide you through setting up your RAG pipeline.[/magenta]"
    )

    console.print("[bold cyan]\n--- 📂 Step 1: Data Source ---[/bold cyan]")
    cwd = os.getcwd()
    data_path_str = typer.prompt(f"Enter the path to the directory with your documents)", default=cwd)
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        console.print(
            f"[bold red]❌ Error: The path '{data_path_str}' is not a valid directory.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print("[bold cyan]\n--- 💾 Step 2: Vector Database ---[/bold cyan]")
    db_path = typer.prompt(
        "Where should the vector database be stored?",
        default=Settings.DEFAULT_PERSIST_DIRECTORY,
    )
    collection = typer.prompt(
        "What is the name for the database collection?",
        default=Settings.DEFAULT_COLLECTION_NAME,
    )

    # --- Embeddings Provider and Model Selection ---
    console.print("[bold cyan]\n--- 🧠 Step 3: Embeddings Model ---[/bold cyan]")
    emb_provider = select_with_arrows(
        "Which embeddings provider do you want to use?",
        list(EMBEDDING_PROVIDERS.keys()),
        default=Settings.HUGGINGFACE
    )
    emb_model = select_with_arrows(
        f"Which embedding model do you want to use for {emb_provider}?",
        EMBEDDING_PROVIDERS[emb_provider],
        default=EMBEDDING_PROVIDERS[emb_provider][0]
    )

    # --- LLM Provider and Model Selection ---
    console.print("[bold cyan]\n--- 🤖 Step 4: Language Model (LLM) ---[/bold cyan]")
    llm_provider = select_with_arrows(
        "Which LLM provider do you want to use?",
        list(LLM_PROVIDERS.keys()),
        default=Settings.OLLAMA
    )
    llm_model = select_with_arrows(
        f"Which LLM do you want to use for {llm_provider}?",
        LLM_PROVIDERS[llm_provider],
        default=LLM_PROVIDERS[llm_provider][0]
    )
    k = typer.prompt(
        "How many documents should be retrieved for context (k)?", type=int, default=5
    )

    console.print("[bold green]\n✅ Configuration complete![/bold green]")

    try:
        console.print("[bold cyan]\n--- ⏳ Step 5: Indexing Documents ---[/bold cyan]")

        should_index = True
        if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
            console.print(f"[yellow]A database seems to exist at '{db_path}'.[/yellow]")
            if not typer.confirm(
                "Do you want to re-index the data? (This will add documents to the existing collection)\nIf you don't want, existing database will be used.",
                default=False,
            ):
                should_index = False

        builder = Builder()
        builder.with_embeddings(emb_provider, model_name=emb_model)
        builder.with_vector_store(
            Settings.CHROMA,
            persist_directory=db_path,
            collection_name=collection,
        )

        if should_index:
            vector_store = builder.build_vector_store()
            vector_store.ingest(data_path=str(data_path))
            vector_store.ingest_code(repos_path=str(data_path))
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

            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold cyan]Waiting for response...[/bold cyan]"),
                transient=True,
                console=console,
            ) as progress:
                task = progress.add_task("", total=None)
                response = rag_pipeline.generate(query)
                progress.update(task, completed=1)

            print_llm_response(response)

    except Exception as e:
        console.print(f"[bold red]❌ An unexpected error occurred: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)
    
@app.command(name="agentic-chat")
def interactive_chat_command():
    """
    Starts a guided, interactive session to configure, index, and chat with your data.
    """
    console.print(
        "[bold magenta]👋 Welcome to the RAGLight Interactive Setup Wizard![/bold magenta]"
    )
    console.print(
        "[magenta]I will guide you through setting up your RAG pipeline.[/magenta]"
    )

    console.print("[bold cyan]\n--- 📂 Step 1: Data Source ---[/bold cyan]")
    cwd = os.getcwd()
    data_path_str = typer.prompt(f"Enter the path to the directory with your documents (default: {cwd})", default=cwd)
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        console.print(
            f"[bold red]❌ Error: The path '{data_path_str}' is not a valid directory.[/bold red]"
        )
        raise typer.Exit(code=1)

    console.print("[bold cyan]\n--- 💾 Step 2: Vector Database ---[/bold cyan]")
    db_path = typer.prompt(
        "Where should the vector database be stored?",
        default=Settings.DEFAULT_PERSIST_DIRECTORY,
    )
    collection = typer.prompt(
        "What is the name for the database collection?",
        default=Settings.DEFAULT_COLLECTION_NAME,
    )

    # --- Embeddings Provider and Model Selection ---
    console.print("[bold cyan]\n--- 🧠 Step 3: Embeddings Model ---[/bold cyan]")
    emb_provider = select_with_arrows(
        "Which embeddings provider do you want to use?",
        list(EMBEDDING_PROVIDERS.keys()),
        default=Settings.HUGGINGFACE
    )
    emb_model = select_with_arrows(
        f"Which embedding model do you want to use for {emb_provider}?",
        EMBEDDING_PROVIDERS[emb_provider],
        default=EMBEDDING_PROVIDERS[emb_provider][0]
    )

    # --- LLM Provider and Model Selection ---
    console.print("[bold cyan]\n--- 🤖 Step 4: Language Model (LLM) ---[/bold cyan]")
    llm_provider = select_with_arrows(
        "Which LLM provider do you want to use?",
        list(LLM_PROVIDERS.keys()),
        default=Settings.OLLAMA
    )
    llm_model = select_with_arrows(
        f"Which LLM do you want to use for {llm_provider}?",
        LLM_PROVIDERS[llm_provider],
        default=LLM_PROVIDERS[llm_provider][0]
    )
    k = typer.prompt(
        "How many documents should be retrieved for context (k)?", type=int, default=5
    )

    console.print("[bold green]\n✅ Configuration complete![/bold green]")

    try:
        console.print("[bold cyan]\n--- ⏳ Step 5: Indexing Documents ---[/bold cyan]")

        should_index = True
        if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
            console.print(f"[yellow]A database seems to exist at '{db_path}'.[/yellow]")
            if not typer.confirm(
                "Do you want to re-index the data? (This will add documents to the existing collection)\nIf you don't want, existing database will be used.",
                default=False,
            ):
                should_index = False

        vector_store_config = VectorStoreConfig(
            embedding_model = emb_model,
            database=Settings.CHROMA,
            persist_directory = db_path,
            provider = emb_provider,
            collection_name = collection
        )

        config = AgenticRAGConfig(
                    provider = llm_provider,
                    model = llm_model,
                    k = k,
                    system_prompt = Settings.DEFAULT_AGENT_PROMPT,
                    max_steps = 4,
                    api_key = Settings.MISTRAL_API_KEY, # os.environ.get('MISTRAL_API_KEY')
                    api_base = Settings.DEFAULT_OLLAMA_CLIENT # If you have a custom client URL
                )

        agenticRag = AgenticRAGPipeline(config, vector_store_config)


        if should_index:
            agenticRag.get_vector_store().ingest(data_path=str(data_path))
            agenticRag.get_vector_store().ingest_code(repos_path=str(data_path))
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


@app.command(name="index", hidden=True)
def index_command(
    data_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Path to the directory containing documents to index.",
        ),
    ],
    db_path: Annotated[
        str, typer.Option("--db-path", "-db")
    ] = Settings.DEFAULT_PERSIST_DIRECTORY,
    collection: Annotated[
        str, typer.Option("--collection", "-c")
    ] = Settings.DEFAULT_COLLECTION_NAME,
    embeddings: Annotated[
        str, typer.Option("--embeddings-model", "-e")
    ] = Settings.DEFAULT_EMBEDDINGS_MODEL,
):
    """(Hidden) A direct command to index a directory for scripting purposes."""
    try:
        vector_store = (
            Builder()
            .with_embeddings(Settings.HUGGINGFACE, model_name=embeddings)
            .with_vector_store(
                Settings.CHROMA, persist_directory=db_path, collection_name=collection
            )
            .build_vector_store()
        )
        vector_store.ingest(data_path=str(data_path))
        vector_store.ingest_code(repos_path=str(data_path))
        console.print(
            f"[bold green]✅ Successfully indexed all documents from {data_path}[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]❌ An error occurred during indexing: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
