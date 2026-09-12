import 'datetime.dart';
import 'parser.dart';

/// Turns a parsed [ToolCall] into a concrete Android dispatch: either an intent
/// to launch (via android_intent_plus) or a native platform call (torch, media
/// keys). Kept pure and free of any plugin import so the whole mapping — the
/// part that is actually easy to get wrong — is unit-tested on the host.
///
/// The launcher that turns these into real side effects lives in `actions.dart`.
sealed class ToolDispatch {
  const ToolDispatch();
}

/// An Android intent to hand to `startActivity`. Mirrors the fields
/// android_intent_plus's `AndroidIntent` takes.
class IntentDispatch extends ToolDispatch {
  const IntentDispatch(
    this.action, {
    this.data,
    this.type,
    this.arguments = const {},
  });

  final String action;
  final String? data;
  final String? type;
  final Map<String, Object?> arguments;
}

/// A call over the `com.offhand/tools` method channel for things no intent can
/// do: the flashlight and media transport keys.
class NativeDispatch extends ToolDispatch {
  const NativeDispatch(this.method, [this.args = const {}]);
  final String method;
  final Map<String, Object?> args;
}

/// A tool name we have no mapping for.
class UnknownDispatch extends ToolDispatch {
  const UnknownDispatch(this.name);
  final String name;
}

// --- Android intent constants (kept here so the mapping reads declaratively) --

const _actionSetAlarm = 'android.intent.action.SET_ALARM';
const _actionSetTimer = 'android.intent.action.SET_TIMER';
const _actionInsert = 'android.intent.action.INSERT';
const _actionSendto = 'android.intent.action.SENDTO';
const _actionSend = 'android.intent.action.SEND';
const _actionPlayFromSearch = 'android.media.action.MEDIA_PLAY_FROM_SEARCH';

const _extraHour = 'android.intent.extra.alarm.HOUR';
const _extraMinutes = 'android.intent.extra.alarm.MINUTES';
const _extraMessage = 'android.intent.extra.alarm.MESSAGE';
const _extraLength = 'android.intent.extra.alarm.LENGTH';
const _extraSubject = 'android.intent.extra.SUBJECT';
const _extraText = 'android.intent.extra.TEXT';
const _extraMediaFocus = 'android.intent.extra.focus';

const _calendarEventsUri = 'content://com.android.calendar/events';

const _settingsActions = {
  'wifi': 'android.settings.WIFI_SETTINGS',
  'bluetooth': 'android.settings.BLUETOOTH_SETTINGS',
  'display': 'android.settings.DISPLAY_SETTINGS',
  'sound': 'android.settings.SOUND_SETTINGS',
  'battery': 'android.settings.BATTERY_SAVER_SETTINGS',
  'location': 'android.settings.LOCATION_SOURCE_SETTINGS',
};

const _transportActions = {'play', 'pause', 'next', 'previous'};

/// Map a tool call to its Android dispatch. [now] is injectable for tests.
ToolDispatch buildDispatch(ToolCall call, {DateTime? now}) {
  String s(String key) => (call.arguments[key] ?? '').toString().trim();

  switch (call.name) {
    case 'set_alarm':
      final clock = parseClockTime(s('time'));
      return IntentDispatch(_actionSetAlarm, arguments: {
        if (clock != null) _extraHour: clock.hour,
        if (clock != null) _extraMinutes: clock.minute,
        if (s('label').isNotEmpty) _extraMessage: s('label'),
      });

    case 'set_timer':
      final minutes = _asNum(call.arguments['duration_minutes']);
      return IntentDispatch(_actionSetTimer, arguments: {
        if (minutes != null) _extraLength: (minutes * 60).round(),
        if (s('label').isNotEmpty) _extraMessage: s('label'),
      });

    case 'create_calendar_event':
      final when = parseDateTime(s('start_datetime'), now: now);
      return _calendarIntent(
        title: s('title'),
        when: when,
        endRaw: s('end_datetime'),
        location: s('location'),
        now: now,
      );

    case 'set_reminder':
      // No true reminder intent exists; a calendar event carries the default
      // notification. Timed → event at that time; untimed → all-day today.
      final when = s('time').isEmpty ? null : parseDateTime(s('time'), now: now);
      return _calendarIntent(
        title: s('text'),
        when: when,
        endRaw: '',
        location: '',
        now: now,
      );

    case 'open_settings':
      final action = _settingsActions[s('panel').toLowerCase()];
      return action == null
          ? UnknownDispatch('open_settings:${s('panel')}')
          : IntentDispatch(action);

    case 'draft_sms':
      return IntentDispatch(
        _actionSendto,
        data: 'smsto:${s('recipient')}',
        arguments: {'sms_body': s('body')},
      );

    case 'draft_email':
      return IntentDispatch(
        _actionSendto,
        data: _mailto(to: s('to'), subject: s('subject'), body: s('body')),
        arguments: {
          if (s('subject').isNotEmpty) _extraSubject: s('subject'),
          if (s('body').isNotEmpty) _extraText: s('body'),
        },
      );

    case 'play_music':
      final query = s('query');
      if (query.isNotEmpty) {
        return IntentDispatch(_actionPlayFromSearch, arguments: {
          _extraMediaFocus: 'vnd.android.cursor.item/*',
          'query': query,
        });
      }
      final action = s('action').toLowerCase();
      return NativeDispatch('mediaKey', {
        'action': _transportActions.contains(action) ? action : 'play',
      });

    case 'toggle_flashlight':
      return NativeDispatch('setTorch', {'on': s('state').toLowerCase() == 'on'});

    case 'create_note':
      return IntentDispatch(
        _actionSend,
        type: 'text/plain',
        arguments: {
          _extraText: s('content'),
          if (s('title').isNotEmpty) _extraSubject: s('title'),
        },
      );

    default:
      return UnknownDispatch(call.name);
  }
}

IntentDispatch _calendarIntent({
  required String title,
  required ParsedDateTime? when,
  required String endRaw,
  required String location,
  DateTime? now,
}) {
  final base = now ?? DateTime.now();
  final args = <String, Object?>{
    if (title.isNotEmpty) 'title': title,
    if (location.isNotEmpty) 'eventLocation': location,
  };

  if (when == null || !when.hasTime) {
    // All-day event on the given day, or today if we could not parse a day.
    final day = when?.dateTime ?? DateTime(base.year, base.month, base.day);
    final start = DateTime(day.year, day.month, day.day);
    args['allDay'] = true;
    args['beginTime'] = start.millisecondsSinceEpoch;
    args['endTime'] = start.add(const Duration(days: 1)).millisecondsSinceEpoch;
  } else {
    final start = when.dateTime;
    final endParsed = endRaw.isEmpty ? null : parseDateTime(endRaw, now: now);
    final end = (endParsed != null && endParsed.dateTime.isAfter(start))
        ? endParsed.dateTime
        : start.add(const Duration(hours: 1));
    args['beginTime'] = start.millisecondsSinceEpoch;
    args['endTime'] = end.millisecondsSinceEpoch;
  }

  return IntentDispatch(_actionInsert, data: _calendarEventsUri, arguments: args);
}

String _mailto({required String to, required String subject, required String body}) {
  // mailto uses standard percent-encoding (RFC 6068): a '+' is a literal plus,
  // so encode with encodeComponent (space -> %20), never encodeQueryComponent
  // (space -> +), or clients render "Hi+there".
  final params = <String>[
    if (subject.isNotEmpty) 'subject=${Uri.encodeComponent(subject)}',
    if (body.isNotEmpty) 'body=${Uri.encodeComponent(body)}',
  ];
  final query = params.isEmpty ? '' : '?${params.join('&')}';
  return 'mailto:${Uri.encodeComponent(to)}$query';
}

num? _asNum(Object? v) {
  if (v is num) return v;
  if (v is String) return num.tryParse(v.trim());
  return null;
}
