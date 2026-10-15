import os
import ast
import re
import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from ...models import Document


class CodeProcessor:
    """A strategy for processing source code files."""

    def process(
        self, file_path: str, chunk_size: int, chunk_overlap: int
    ) -> Dict[str, List[Document]]:
        """
        Processes a single source code file. It extracts class signatures
        and splits the full code into chunks.

        Returns:
            A dictionary containing two lists of documents:
            - "chunks": The full code split into chunks.
            - "classes": Documents representing only the class signatures.
        """
        language = self._get_language_from_extension(
            os.path.splitext(file_path)[1][1:].lower()
        )
        if not language:
            return {"chunks": [], "classes": []}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            if not code.strip():
                return {"chunks": [], "classes": []}

            class_signatures = self._extract_class_signatures(code, language)
            class_docs = [
                Document(page_content=sig, metadata={"source": file_path})
                for sig in class_signatures
            ]

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            split_documents = splitter.create_documents([code])

            class_names_str = ", ".join(class_signatures)
            code_chunks = []
            for chunk in split_documents:
                metadata = dict(chunk.metadata)
                metadata["source"] = file_path
                if class_names_str:
                    metadata["classes"] = class_names_str
                code_chunks.append(
                    Document(page_content=chunk.page_content, metadata=metadata)
                )

            return {"chunks": code_chunks, "classes": class_docs}
        except Exception as e:
            logging.error(f"Failed to process code file {file_path}. Error: {e}")
            return {"chunks": [], "classes": []}

    def _extract_class_signatures(self, code: str, language: Language) -> List[str]:
        """Dispatcher method to extract class signatures based on language."""
        if language == Language.PYTHON:
            return self._extract_python_class_signatures(code)
        elif language in {
            Language.JS,
            Language.TS,
            Language.JAVA,
            Language.CPP,
            Language.CSHARP,
        }:
            return self._extract_class_signatures_with_regex(code, language)
        return []

    def _extract_python_class_signatures(self, code: str) -> List[str]:
        """Extracts class signatures from Python code using AST."""
        try:
            tree = ast.parse(code)
            class_signatures = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                    class_signature = f"class {node.name}({', '.join(bases)})"
                    class_signatures.append(class_signature)
            return class_signatures
        except SyntaxError:
            return []

    def _extract_class_signatures_with_regex(
        self, code: str, language: Language
    ) -> List[str]:
        """Extracts class signatures using regex for various languages."""
        patterns = {
            Language.JAVA: r"class\s+(\w+)\s*(?:extends\s+\w+)?\s*(?:implements\s+[\w,\s]+)?",
            Language.JS: r"class\s+(\w+)(?:\s+extends\s+\w+)?",
            Language.TS: r"class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?",
            Language.CPP: r"class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)*",
            Language.CSHARP: r"class\s+(\w+)(?:\s*:\s*[\w,\s]+)?",
        }
        pattern = patterns.get(language)
        if not pattern:
            return []

        matches = re.findall(pattern, code)
        full_signatures = [f"class {match}" for match in matches]
        return full_signatures

    def _get_language_from_extension(self, extension: str) -> Language | None:
        """Maps a file extension to a Language enum."""
        extension_to_language = {
            "py": Language.PYTHON,
            "js": Language.JS,
            "ts": Language.TS,
            "java": Language.JAVA,
            "cpp": Language.CPP,
            "go": Language.GO,
            "php": Language.PHP,
            "rb": Language.RUBY,
            "rs": Language.RUST,
            "scala": Language.SCALA,
            "swift": Language.SWIFT,
            "md": Language.MARKDOWN,
            "html": Language.HTML,
            "sol": Language.SOL,
            "cs": Language.CSHARP,
            "c": Language.C,
            "lua": Language.LUA,
            "pl": Language.PERL,
            "hs": Language.HASKELL,
        }
        return extension_to_language.get(extension)
