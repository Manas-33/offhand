import 'package:android_intent_plus/android_intent.dart';
import 'package:flutter/services.dart';

import '../bridge/native_tools.dart';
import 'dispatch.dart';
import 'parser.dart';

final NativeTools _native = NativeTools();

/// A one-line, plain-English summary of what a tool call will do, shown as the
/// heading on the approval sheet.
String describeAction(ToolCall call) {
  String a(String k) => call.arguments[k]?.toString().trim() ?? '';
  switch (call.name) {
    case 'set_alarm':
      final label = a('label');
      return 'Set an alarm for ${a('time')}${label.isEmpty ? '' : ' — $label'}';
    case 'set_timer':
      final label = a('label');
      return 'Start a ${_plainNum(a('duration_minutes'))} min timer'
          '${label.isEmpty ? '' : ' — $label'}';
    case 'create_calendar_event':
      final start = a('start_datetime');
      return 'Add event "${a('title')}"${start.isEmpty ? '' : ' at $start'}';
    case 'set_reminder':
      final time = a('time');
      return 'Remind you: ${a('text')}${time.isEmpty ? '' : ' at $time'}';
    case 'open_settings':
      return 'Open ${a('panel')} settings';
    case 'draft_sms':
      return 'Draft a text to ${a('recipient')}';
    case 'draft_email':
      return 'Draft an email to ${a('to')}';
    case 'play_music':
      final q = a('query');
      final action = a('action').isEmpty ? 'play' : a('action');
      return q.isNotEmpty
          ? 'Play $q'
          : '${action[0].toUpperCase()}${action.substring(1)} music';
    case 'toggle_flashlight':
      return 'Turn the flashlight ${a('state')}';
    case 'create_note':
      final t = a('title');
      return t.isEmpty ? 'Create a note' : 'Create note "$t"';
    default:
      return '${call.name}(${call.arguments})';
  }
}

/// The content the user should actually read before approving — the message or
/// note body — or null for tools with nothing to review. Shown in a box on the
/// approval sheet so a draft is never sent sight-unseen.
String? actionDetail(ToolCall call) {
  String a(String k) => call.arguments[k]?.toString().trim() ?? '';
  switch (call.name) {
    case 'draft_sms':
      final body = a('body');
      return body.isEmpty ? null : body;
    case 'draft_email':
      final subject = a('subject');
      final body = a('body');
      if (subject.isEmpty && body.isEmpty) return null;
      return [
        if (subject.isNotEmpty) 'Subject: $subject',
        if (body.isNotEmpty) body,
      ].join('\n\n');
    case 'create_note':
      final content = a('content');
      return content.isEmpty ? null : content;
    default:
      return null;
  }
}

/// Execute an approved tool call. Intent-backed tools launch an Android intent;
/// the flashlight and media keys go over the native channel. Returns a short
/// result message for the snackbar.
Future<String> executeAction(ToolCall call) async {
  final dispatch = buildDispatch(call);
  switch (dispatch) {
    case IntentDispatch():
      try {
        final intent = AndroidIntent(
          action: dispatch.action,
          data: dispatch.data,
          type: dispatch.type,
          arguments: dispatch.arguments.isEmpty
              ? null
              : Map<String, dynamic>.from(dispatch.arguments),
        );
        await intent.launch();
        return _successMessage(call.name);
      } on PlatformException catch (e) {
        return 'Could not open an app for ${call.name} (${e.code})';
      } catch (_) {
        return 'Could not complete ${call.name}';
      }
    case NativeDispatch():
      return _runNative(dispatch);
    case UnknownDispatch():
      return 'No handler for ${dispatch.name}';
  }
}

Future<String> _runNative(NativeDispatch d) async {
  try {
    switch (d.method) {
      case 'setTorch':
        final on = d.args['on'] == true;
        final ok = await _native.setTorch(on);
        return ok ? 'Flashlight ${on ? 'on' : 'off'}' : 'No flashlight available';
      case 'mediaKey':
        final action = d.args['action']?.toString() ?? 'play';
        await _native.mediaKey(action);
        return 'Media: $action';
      default:
        return 'No handler for ${d.method}';
    }
  } on PlatformException catch (e) {
    return 'Native action failed (${e.code})';
  }
}

String _successMessage(String name) {
  switch (name) {
    case 'draft_sms':
      return 'Opened a text draft';
    case 'draft_email':
      return 'Opened an email draft';
    case 'create_note':
      return 'Shared to notes';
    case 'set_alarm':
      return 'Opened alarm';
    case 'set_timer':
      return 'Opened timer';
    case 'create_calendar_event':
    case 'set_reminder':
      return 'Opened calendar event';
    case 'open_settings':
      return 'Opened settings';
    case 'play_music':
      return 'Playing';
    default:
      return 'Done';
  }
}

/// Format a possibly-decimal minute count as a plain integer when it is whole.
String _plainNum(String raw) {
  final n = num.tryParse(raw);
  if (n == null) return raw.isEmpty ? '?' : raw;
  return n == n.roundToDouble() ? n.toInt().toString() : n.toString();
}
