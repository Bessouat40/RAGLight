"""Unit tests for the public integration registries."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest

from raglight.integrations import IntegrationRegistry, default_registries


class ExampleComponent:
    def __init__(self, *, value: str) -> None:
        self.value = value


class IntegrationRegistryTests(unittest.TestCase):
    def test_custom_factory_can_be_registered_and_created(self):
        registry = IntegrationRegistry("example")
        registry.register("Custom", ExampleComponent)

        component = registry.create(" custom ", value="configured")

        self.assertIsInstance(component, ExampleComponent)
        self.assertEqual("configured", component.value)
        self.assertEqual(("custom",), registry.names)

    def test_duplicate_registration_requires_explicit_replacement(self):
        registry = IntegrationRegistry("example")
        registry.register("custom", ExampleComponent)

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("custom", ExampleComponent)

        registry.register("custom", lambda **options: options, replace=True)
        self.assertEqual({"value": "new"}, registry.create("custom", value="new"))

    def test_unknown_name_lists_available_integrations(self):
        registry = IntegrationRegistry("LLM")
        registry.register("known", ExampleComponent)

        with self.assertRaisesRegex(
            LookupError, "Unknown LLM integration 'missing'.*known"
        ):
            registry.create("missing")

    def test_invalid_names_are_rejected(self):
        registry = IntegrationRegistry("example")

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            registry.register("   ", ExampleComponent)

    def test_lazy_factory_is_resolved_when_created(self):
        registry = IntegrationRegistry("example")
        registry.register_lazy("namespace", "types", "SimpleNamespace")

        component = registry.create("namespace", value="lazy")

        self.assertEqual("lazy", component.value)

    def test_registered_factory_must_be_callable(self):
        registry = IntegrationRegistry("example")

        with self.assertRaisesRegex(TypeError, "must be callable"):
            registry.register("invalid", object())

    def test_builtin_registries_are_complete(self):
        self.assertEqual(
            ("bedrock", "gemini", "lmstudio", "mistral", "ollama", "openai"),
            default_registries.llms.names,
        )
        self.assertEqual(
            ("bedrock", "gemini", "huggingface", "ollama", "openai"),
            default_registries.embeddings.names,
        )
        self.assertEqual(("chroma", "qdrant"), default_registries.vector_stores.names)
        self.assertEqual(("huggingface",), default_registries.rerankers.names)
        self.assertEqual(
            (
                "c",
                "cpp",
                "cs",
                "go",
                "hs",
                "html",
                "java",
                "js",
                "lua",
                "md",
                "pdf",
                "php",
                "pl",
                "py",
                "rb",
                "rs",
                "scala",
                "sol",
                "swift",
                "ts",
                "txt",
            ),
            default_registries.document_processors.names,
        )

    def test_importing_registries_does_not_import_provider_modules(self):
        script = textwrap.dedent("""
            import sys
            from raglight.integrations import default_registries

            assert default_registries.llms.names
            loaded = [
                name for name in sys.modules
                if name.startswith("raglight.integrations.")
            ]
            raise SystemExit(f"unexpected imports: {loaded}" if loaded else 0)
            """)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)


if __name__ == "__main__":
    unittest.main()
