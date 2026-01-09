import nltk
import typer
from pathlib import Path
import logging
import os
import questionary
import shutil
from typing import List

from raglight.rag.builder import Builder
from raglight.config.settings import Settings
from raglight.rag.rag import RAG
from typing_extensions import Annotated

from rich.console import Console
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

custom_style = questionary.Style(
    [
        ("answer", "bold ansicyan"),
    ]
)


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
    return questionary.select(message, choices=choices, default=default).ask()


def prompt_github_sources() -> List[GitHubSource]:
    console.print(
        "[bold cyan]\n--- 🧩 Step 1.25: GitHub Knowledge Source ---[/bold cyan]"
    )
    github_sources: List[GitHubSource] = []
    if typer.confirm(
        "Do you want to add a GitHub repository as a knowledge source?",
        default=False,
    ):
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
    data_path_str = typer.prompt(
        f"Enter the path to the directory with your documents)", default=cwd
    )
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        console.print(
            f"[bold red]❌ Error: The path '{data_path_str}' is not a valid directory.[/bold red]"
        )
        raise typer.Exit(code=1)

    # Configure ignore folders
    console.print(
        "[bold cyan]\n--- 🚫 Step 1.5: Ignore Folders Configuration ---[/bold cyan]"
    )
    console.print(
        "[yellow]By default, the following folders will be ignored during indexing:[/yellow]"
    )
    default_ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS
    for folder in default_ignore_folders:
        console.print(f"  • {folder}")

    if typer.confirm(
        "Do you want to customize the ignore folders list?", default=False
    ):
        ignore_folders = []
        console.print(
            "[cyan]Enter folder names to ignore (one per line, press Enter twice to finish):[/cyan]"
        )
        console.print(
            "[yellow]Leave empty to use default list, or type 'default' to use default list[/yellow]"
        )

        while True:
            folder = input("Folder to ignore (or Enter to finish): ").strip()
            if not folder:
                break
            if folder.lower() == "default":
                ignore_folders = default_ignore_folders.copy()
                break
            ignore_folders.append(folder)

        if not ignore_folders:
            ignore_folders = default_ignore_folders.copy()
    else:
        ignore_folders = default_ignore_folders.copy()

    console.print(
        f"[green]✅ Will ignore {len(ignore_folders)} folders during indexing[/green]"
    )

    github_sources = prompt_github_sources()

    console.print("[bold cyan]\n--- 💾 Step 2: Vector Database ---[/bold cyan]")
    db_path = typer.prompt(
        "Where should the vector database be stored?",
        default=Settings.DEFAULT_PERSIST_DIRECTORY,
    )
    collection = typer.prompt(
        "What is the name for the database collection?",
        default=Settings.DEFAULT_COLLECTION_NAME,
    )

    console.print("[bold blue]\n--- 🧠 Step 3: Embeddings Model ---[/bold blue]")
    emb_provider = questionary.select(
        "Which embeddings provider do you want to use?",
        choices=[
            Settings.HUGGINGFACE,
            Settings.OLLAMA,
            Settings.OPENAI,
            Settings.GOOGLE_GEMINI,
        ],
        default=Settings.HUGGINGFACE,
        style=custom_style,
    ).ask()

    default_api_base = None
    if emb_provider == Settings.OLLAMA:
        default_api_base = Settings.DEFAULT_OLLAMA_CLIENT
    elif emb_provider == Settings.OPENAI:
        default_api_base = Settings.DEFAULT_OPENAI_CLIENT
    elif emb_provider == Settings.GOOGLE_GEMINI:
        default_api_base = Settings.DEFAULT_GOOGLE_CLIENT

    embeddings_base_url = RichPrompt.ask(
        "[bold]What is your base URL for the embeddings provider? (Not needed for HuggingFace)[/bold]",
        default=default_api_base,
    )
    emb_model = RichPrompt.ask(
        "[bold]Which embedding model do you want to use?[/bold]",
        default=Settings.DEFAULT_EMBEDDINGS_MODEL,
    )

    console.print("[bold blue]\n--- 🤖 Step 4: Language Model (LLM) ---[/bold blue]")
    llm_provider = questionary.select(
        "Which LLM provider do you want to use?",
        choices=[
            Settings.OLLAMA,
            Settings.MISTRAL,
            Settings.OPENAI,
            Settings.LMSTUDIO,
            Settings.GOOGLE_GEMINI,
        ],
        default=Settings.OLLAMA,
        style=custom_style,
    ).ask()

    llm_default_api_base = None
    if llm_provider == Settings.OLLAMA:
        llm_default_api_base = Settings.DEFAULT_OLLAMA_CLIENT
    elif llm_provider == Settings.OPENAI:
        llm_default_api_base = Settings.DEFAULT_OPENAI_CLIENT
    elif llm_provider == Settings.LMSTUDIO:
        llm_default_api_base = Settings.DEFAULT_LMSTUDIO_CLIENT
    elif llm_provider == Settings.GOOGLE_GEMINI:
        llm_default_api_base = Settings.DEFAULT_GOOGLE_CLIENT

    llm_base_url = RichPrompt.ask(
        "[bold]What is your base URL for the LLM provider? (Not needed for Mistral)[/bold]",
        default=llm_default_api_base,
    )

    llm_model = RichPrompt.ask(
        "[bold]Which LLM do you want to use?[/bold]",
        default=Settings.DEFAULT_LLM,
    )
    k = questionary.select(
        "How many documents should be retrieved for context (k)?",
        choices=["5", "10", "15"],
        default=str(Settings.DEFAULT_K),
        style=custom_style,
    ).ask()
    k = int(k)

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
        builder.with_embeddings(
            emb_provider, model_name=emb_model, api_base=embeddings_base_url
        )
        builder.with_vector_store(
            Settings.CHROMA,
            persist_directory=db_path,
            collection_name=collection,
        )

        if should_index:
            vector_store = builder.build_vector_store()
            vector_store.ingest(data_path=str(data_path), ignore_folders=ignore_folders)
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
    data_path_str = typer.prompt(
        f"Enter the path to the directory with your documents)", default=cwd
    )
    data_path = Path(data_path_str)
    if not data_path.is_dir():
        console.print(
            f"[bold red]❌ Error: The path '{data_path_str}' is not a valid directory.[/bold red]"
        )
        raise typer.Exit(code=1)

    # Configure ignore folders
    console.print(
        "[bold cyan]\n--- 🚫 Step 1.5: Ignore Folders Configuration ---[/bold cyan]"
    )
    console.print(
        "[yellow]By default, the following folders will be ignored during indexing:[/yellow]"
    )
    default_ignore_folders = Settings.DEFAULT_IGNORE_FOLDERS
    for folder in default_ignore_folders:
        console.print(f"  • {folder}")

    if typer.confirm(
        "Do you want to customize the ignore folders list?", default=False
    ):
        ignore_folders = []
        console.print(
            "[cyan]Enter folder names to ignore (one per line, press Enter twice to finish):[/cyan]"
        )
        console.print(
            "[yellow]Leave empty to use default list, or type 'default' to use default list[/yellow]"
        )

        while True:
            folder = input("Folder to ignore (or Enter to finish): ").strip()
            if not folder:
                break
            if folder.lower() == "default":
                ignore_folders = default_ignore_folders.copy()
                break
            ignore_folders.append(folder)

        if not ignore_folders:
            ignore_folders = default_ignore_folders.copy()
    else:
        ignore_folders = default_ignore_folders.copy()

    console.print(
        f"[green]✅ Will ignore {len(ignore_folders)} folders during indexing[/green]"
    )

    github_sources = prompt_github_sources()

    console.print("[bold cyan]\n--- 💾 Step 2: Vector Database ---[/bold cyan]")
    db_path = typer.prompt(
        "Where should the vector database be stored?",
        default=Settings.DEFAULT_PERSIST_DIRECTORY,
    )
    collection = typer.prompt(
        "What is the name for the database collection?",
        default=Settings.DEFAULT_COLLECTION_NAME,
    )

    console.print("[bold blue]\n--- 🧠 Step 3: Embeddings Model ---[/bold blue]")
    emb_provider = questionary.select(
        "Which embeddings provider do you want to use?",
        choices=[Settings.HUGGINGFACE, Settings.OLLAMA, Settings.OPENAI],
        default=Settings.HUGGINGFACE,
        style=custom_style,
    ).ask()

    default_api_base = None
    if emb_provider == Settings.OLLAMA:
        default_api_base = Settings.DEFAULT_OLLAMA_CLIENT
    elif emb_provider == Settings.OPENAI:
        default_api_base = Settings.DEFAULT_OPENAI_CLIENT

    embeddings_base_url = RichPrompt.ask(
        "[bold]What is your base URL for the embeddings provider? (Not needed for HuggingFace)[/bold]",
        default=default_api_base,
    )
    emb_model = RichPrompt.ask(
        "[bold]Which embedding model do you want to use?[/bold]",
        default=Settings.DEFAULT_EMBEDDINGS_MODEL,
    )

    console.print("[bold blue]\n--- 🤖 Step 4: Language Model (LLM) ---[/bold blue]")
    llm_provider = questionary.select(
        "Which LLM provider do you want to use?",
        choices=[Settings.OLLAMA, Settings.MISTRAL, Settings.OPENAI, Settings.LMSTUDIO],
        default=Settings.OLLAMA,
        style=custom_style,
    ).ask()

    llm_default_api_base = None
    if llm_provider == Settings.OLLAMA:
        llm_default_api_base = Settings.DEFAULT_OLLAMA_CLIENT
    elif llm_provider == Settings.OPENAI:
        llm_default_api_base = Settings.DEFAULT_OPENAI_CLIENT
    elif llm_provider == Settings.LMSTUDIO:
        llm_default_api_base = Settings.DEFAULT_LMSTUDIO_CLIENT

    llm_base_url = RichPrompt.ask(
        "[bold]What is your base URL for the LLM provider? (Not needed for Mistral)[/bold]",
        default=llm_default_api_base,
    )

    llm_model = RichPrompt.ask(
        "[bold]Which LLM do you want to use?[/bold]",
        default=Settings.DEFAULT_LLM,
    )
    k = questionary.select(
        "How many documents should be retrieved for context (k)?",
        choices=["5", "10", "15"],
        default=str(Settings.DEFAULT_K),
        style=custom_style,
    ).ask()
    k = int(k)

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
            embedding_model=emb_model,
            api_base=embeddings_base_url,
            database=Settings.CHROMA,
            persist_directory=db_path,
            provider=emb_provider,
            collection_name=collection,
        )

        config = AgenticRAGConfig(
            provider=llm_provider,
            model=llm_model,
            k=k,
            system_prompt=Settings.DEFAULT_AGENT_PROMPT,
            max_steps=4,
            api_key=Settings.MISTRAL_API_KEY,  # os.environ.get('MISTRAL_API_KEY')
            api_base=llm_base_url,
        )

        agenticRag = AgenticRAGPipeline(config, vector_store_config)

        if should_index:
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


if __name__ == "__main__":
    app()
