import os
import tempfile
import unittest

from raglight.integrations.documents import CodeProcessor, TextProcessor
from raglight.models import Document


class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()

    def _write_tmp(self, content, suffix=".txt"):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        f.write(content)
        f.close()
        return f.name

    def tearDown(self):
        pass  # files deleted per-test

    def test_returns_correct_keys(self):
        path = self._write_tmp("Hello world. " * 20)
        try:
            result = self.processor.process(path, chunk_size=50, chunk_overlap=0)
            self.assertIn("chunks", result)
            self.assertIn("classes", result)
        finally:
            os.unlink(path)

    def test_chunks_are_not_empty(self):
        path = self._write_tmp("Hello world. " * 50)
        try:
            result = self.processor.process(path, chunk_size=50, chunk_overlap=0)
            self.assertGreater(len(result["chunks"]), 0)
            self.assertTrue(
                all(isinstance(chunk, Document) for chunk in result["chunks"])
            )
            self.assertEqual(result["classes"], [])
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty_chunks(self):
        path = self._write_tmp("")
        try:
            result = self.processor.process(path, chunk_size=100, chunk_overlap=0)
            self.assertEqual(result["chunks"], [])
            self.assertEqual(result["classes"], [])
        finally:
            os.unlink(path)

    def test_file_errors_are_reported_to_the_ingestion_orchestrator(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.process(
                "/missing/raglight-document.txt",
                chunk_size=100,
                chunk_overlap=0,
            )


class TestCodeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = CodeProcessor()

    def _write_tmp_py(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_returns_correct_keys(self):
        path = self._write_tmp_py("x = 1\n")
        try:
            result = self.processor.process(path, chunk_size=500, chunk_overlap=0)
            self.assertIn("chunks", result)
            self.assertIn("classes", result)
        finally:
            os.unlink(path)

    def test_extracts_class_signatures(self):
        code = "class Foo:\n    pass\n\nclass Bar(Foo):\n    pass\n"
        path = self._write_tmp_py(code)
        try:
            result = self.processor.process(path, chunk_size=1000, chunk_overlap=0)
            class_contents = [doc.page_content for doc in result["classes"]]
            self.assertTrue(all(isinstance(doc, Document) for doc in result["classes"]))
            self.assertTrue(all(isinstance(doc, Document) for doc in result["chunks"]))
            self.assertTrue(any("Foo" in c for c in class_contents))
            self.assertTrue(any("Bar" in c for c in class_contents))
        finally:
            os.unlink(path)

    def test_no_classes_returns_empty_classes_list(self):
        path = self._write_tmp_py("def hello():\n    return 'hi'\n")
        try:
            result = self.processor.process(path, chunk_size=500, chunk_overlap=0)
            self.assertEqual(result["classes"], [])
        finally:
            os.unlink(path)

    def test_unsupported_extension_returns_empty(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False)
        f.write("some content")
        f.close()
        try:
            result = self.processor.process(f.name, chunk_size=500, chunk_overlap=0)
            self.assertEqual(result, {"chunks": [], "classes": []})
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
