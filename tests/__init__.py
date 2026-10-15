"""
Stub langgraph.prebuilt.tool_node so that langchain==1.2.0 imports succeed
even though langgraph==1.0.5 does not expose the expected symbols.
Loaded automatically by both unittest and pytest before any test module.
"""

import os
import sys
from unittest.mock import MagicMock

# Ensure src/ is first in sys.path so the local raglight package
# takes precedence over any installed version in site-packages.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

_fake = MagicMock()
_fake.__name__ = "langgraph.prebuilt.tool_node"
_fake.__loader__ = None
_fake.__package__ = "langgraph.prebuilt"
_fake.__spec__ = None
sys.modules["langgraph.prebuilt.tool_node"] = _fake

import langgraph.prebuilt as _lgp  # noqa: E402

_lgp.InjectedState = MagicMock()
_lgp.InjectedStore = MagicMock()
_lgp.ToolRuntime = MagicMock()
