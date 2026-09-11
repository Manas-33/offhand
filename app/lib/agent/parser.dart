import 'dart:convert';

/// A parsed tool call: the function name and its arguments.
class ToolCall {
  ToolCall(this.name, this.arguments);
  final String name;
  final Map<String, dynamic> arguments;
}

final RegExp _toolCallRe =
    RegExp(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', dotAll: true);

/// Extract tool calls from raw model text.
///
/// Mirrors the strict path of the eval harness's parse.py: prefer well-formed
/// `<tool_call>{...}</tool_call>` blocks, then fall back to the first bare JSON
/// object that carries a "name".
List<ToolCall> parseToolCalls(String text) {
  final calls = <ToolCall>[];
  for (final match in _toolCallRe.allMatches(text)) {
    final obj = _tryJson(match.group(1)!);
    if (obj != null && obj['name'] is String) {
      calls.add(_normalize(obj));
    }
  }
  if (calls.isEmpty) {
    final obj = _firstJsonObject(text);
    if (obj != null && obj['name'] is String) {
      calls.add(_normalize(obj));
    }
  }
  return calls;
}

ToolCall _normalize(Map<String, dynamic> obj) {
  dynamic args = obj['arguments'] ?? obj['parameters'] ?? <String, dynamic>{};
  if (args is String) args = _tryJson(args) ?? <String, dynamic>{};
  if (args is! Map) args = <String, dynamic>{};
  return ToolCall(obj['name'] as String, Map<String, dynamic>.from(args));
}

Map<String, dynamic>? _tryJson(String source) {
  try {
    final value = jsonDecode(source);
    return value is Map ? Map<String, dynamic>.from(value) : null;
  } catch (_) {
    return null;
  }
}

/// Scan for the first balanced `{...}` that parses as a JSON object.
Map<String, dynamic>? _firstJsonObject(String text) {
  int depth = 0;
  int start = -1;
  for (int i = 0; i < text.length; i++) {
    final ch = text[i];
    if (ch == '{') {
      if (depth == 0) start = i;
      depth++;
    } else if (ch == '}') {
      if (depth > 0) {
        depth--;
        if (depth == 0 && start >= 0) {
          final obj = _tryJson(text.substring(start, i + 1));
          if (obj != null) return obj;
        }
      }
    }
  }
  return null;
}
