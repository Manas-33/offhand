import 'package:flutter_test/flutter_test.dart';

import 'package:offhand/agent/actions.dart';
import 'package:offhand/agent/parser.dart';

void main() {
  test('parses a wrapped tool call', () {
    final calls = parseToolCalls(
        '<tool_call> {"name":"set_alarm","arguments":{"time":"07:00"}} </tool_call>');
    expect(calls, hasLength(1));
    expect(calls.first.name, 'set_alarm');
    expect(calls.first.arguments['time'], '07:00');
  });

  test('recovers a bare json object with a name', () {
    final calls = parseToolCalls(
        'Sure! {"name":"toggle_flashlight","arguments":{"state":"on"}}');
    expect(calls.first.name, 'toggle_flashlight');
    expect(calls.first.arguments['state'], 'on');
  });

  test('returns empty for plain text', () {
    expect(parseToolCalls('The capital of France is Paris.'), isEmpty);
  });

  test('describeAction gives a plain-English summary', () {
    final call = ToolCall('set_alarm', {'time': '07:00'});
    expect(describeAction(call), 'Set an alarm for 07:00');
  });
}
