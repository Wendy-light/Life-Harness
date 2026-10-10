"""SGLang adapter for AppWorld's xLAM JSON-array tool-call protocol.

Non-streaming behavior follows experiments/code/common/vllm_plugins/
xlam_tool_call_parser.py: parse a complete JSON array, preserve ordinary text
and malformed outputs, and do not silently drop unknown function names.
"""
import json

from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser


class XLAMDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self._stream_done = False

    def has_tool_call(self, text):
        return text.lstrip().startswith("[")

    def supports_structural_tag(self):
        return False

    def structure_info(self):
        raise NotImplementedError("xLAM uses JSON arrays, not structural tags")

    def detect_and_parse(self, text, tools):
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text)
        try:
            values = json.loads(text)
            calls = [ToolCallItem(tool_index=i, name=value["name"],
                                  parameters=json.dumps(value["arguments"]))
                     for i, value in enumerate(values)]
            return StreamingParseResult(calls=calls)
        except (ValueError, TypeError, KeyError):
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text, tools):
        # Buffer arrays until complete; do not emit partial argument JSON.
        if self._stream_done:
            return StreamingParseResult(normal_text=new_text)
        self._buffer += new_text
        if not self._buffer.strip():
            return StreamingParseResult()
        if not self.has_tool_call(self._buffer):
            result = StreamingParseResult(normal_text=self._buffer)
            self._buffer = ""
            self._stream_done = True
            return result
        try:
            json.loads(self._buffer)
        except ValueError:
            return StreamingParseResult()
        result = self.detect_and_parse(self._buffer, tools)
        self._buffer = ""
        self._stream_done = True
        return result


FunctionCallParser.ToolCallParserEnum["xlam"] = XLAMDetector
