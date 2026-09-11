import 'parser.dart';

/// A plain-English summary of what a tool call will do, shown on the approval
/// sheet so the user knows what they are approving.
String describeAction(ToolCall call) {
  String arg(String key) => call.arguments[key]?.toString() ?? '';
  switch (call.name) {
    case 'set_alarm':
      return 'Set an alarm for ${arg('time')}';
    case 'set_timer':
      return 'Start a ${arg('duration_minutes')} minute timer';
    case 'create_calendar_event':
      return 'Add calendar event: ${arg('title')}';
    case 'open_settings':
      return 'Open ${arg('panel')} settings';
    case 'draft_sms':
      return 'Draft a text to ${arg('recipient')}';
    case 'draft_email':
      return 'Draft an email to ${arg('to')}';
    case 'play_music':
      return arg('query').isNotEmpty ? 'Play ${arg('query')}' : 'Music: ${arg('action')}';
    case 'toggle_flashlight':
      return 'Turn the flashlight ${arg('state')}';
    case 'create_note':
      return 'Create a note: ${arg('content')}';
    case 'set_reminder':
      return 'Set a reminder: ${arg('text')}';
    default:
      return '${call.name}(${call.arguments})';
  }
}

/// Execute an approved tool call.
///
/// Stub for now: the real implementation dispatches an Android intent
/// (via android_intent_plus) per tool. Returns a short result message.
Future<String> executeAction(ToolCall call) async {
  // TODO: dispatch to the matching Android intent.
  return 'Executed ${call.name}';
}
