from contextlib import contextmanager
import subprocess
import sys
import textwrap
import unittest


class FakeObservation:
    def update(self, **values):
        self.values = values


class FakeLangfuseClient:
    def __init__(self):
        self.calls = []

    @contextmanager
    def start_as_current_observation(self, **values):
        self.calls.append(values)
        yield FakeObservation()

    def flush(self):
        self.flushed = True


class LangfuseTracerTests(unittest.TestCase):
    def test_adapter_maps_raglight_observations_to_the_langfuse_sdk(self):
        from raglight.integrations.langfuse import LangfuseTracer

        client = FakeLangfuseClient()
        tracer = LangfuseTracer(client=client)

        with tracer.observe(
            name="raglight.rag",
            as_type="chain",
            input={"question": "Hello"},
            metadata={"component": "classic"},
            trace_id="8eb7ae39-5f19-4f57-b99a-362f7dc4155e",
        ):
            pass

        self.assertEqual(
            {
                "name": "raglight.rag",
                "as_type": "chain",
                "input": {"question": "Hello"},
                "metadata": {"component": "classic"},
                "trace_context": {"trace_id": "8eb7ae395f194f57b99a362f7dc4155e"},
            },
            client.calls[0],
        )

    def test_flush_is_explicit_for_short_lived_processes(self):
        from raglight.integrations.langfuse import LangfuseTracer

        client = FakeLangfuseClient()
        tracer = LangfuseTracer(client=client)

        tracer.flush()

        self.assertTrue(client.flushed)

    def test_missing_sdk_names_the_langfuse_extra(self):
        script = textwrap.dedent("""
            import builtins

            original_import = builtins.__import__
            def blocked_import(name, *args, **kwargs):
                if name == "langfuse" or name.startswith("langfuse."):
                    error = ModuleNotFoundError("No module named 'langfuse'")
                    error.name = "langfuse"
                    raise error
                return original_import(name, *args, **kwargs)

            builtins.__import__ = blocked_import
            from raglight.integrations.langfuse import LangfuseTracer
            try:
                LangfuseTracer()
            except ImportError as error:
                message = str(error)
                raise SystemExit(0 if "raglight[langfuse]" in message else message)
            raise SystemExit("Langfuse construction unexpectedly succeeded")
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
