import 'package:flutter_test/flutter_test.dart';

import 'package:offhand/agent/dispatch.dart';
import 'package:offhand/agent/parser.dart';

// The literal Android extra/action keys the mapping emits.
const _setAlarm = 'android.intent.action.SET_ALARM';
const _setTimer = 'android.intent.action.SET_TIMER';
const _insert = 'android.intent.action.INSERT';
const _sendto = 'android.intent.action.SENDTO';
const _send = 'android.intent.action.SEND';
const _playFromSearch = 'android.media.action.MEDIA_PLAY_FROM_SEARCH';

void main() {
  final now = DateTime(2026, 9, 12, 8, 0);

  IntentDispatch intent(String name, Map<String, dynamic> args) {
    final d = buildDispatch(ToolCall(name, args), now: now);
    expect(d, isA<IntentDispatch>());
    return d as IntentDispatch;
  }

  NativeDispatch native(String name, Map<String, dynamic> args) {
    final d = buildDispatch(ToolCall(name, args), now: now);
    expect(d, isA<NativeDispatch>());
    return d as NativeDispatch;
  }

  group('set_alarm', () {
    test('HH:MM', () {
      final d = intent('set_alarm', {'time': '07:00', 'label': 'Wake up'});
      expect(d.action, _setAlarm);
      expect(d.arguments['android.intent.extra.alarm.HOUR'], 7);
      expect(d.arguments['android.intent.extra.alarm.MINUTES'], 0);
      expect(d.arguments['android.intent.extra.alarm.MESSAGE'], 'Wake up');
    });

    test('am/pm', () {
      final d = intent('set_alarm', {'time': '6:30 pm'});
      expect(d.arguments['android.intent.extra.alarm.HOUR'], 18);
      expect(d.arguments['android.intent.extra.alarm.MINUTES'], 30);
    });
  });

  group('set_timer', () {
    test('minutes to seconds', () {
      final d = intent('set_timer', {'duration_minutes': 10});
      expect(d.action, _setTimer);
      expect(d.arguments['android.intent.extra.alarm.LENGTH'], 600);
    });

    test('string minutes', () {
      final d = intent('set_timer', {'duration_minutes': '5'});
      expect(d.arguments['android.intent.extra.alarm.LENGTH'], 300);
    });
  });

  group('calendar and reminder', () {
    test('timed calendar event', () {
      final d = intent('create_calendar_event',
          {'title': 'Team sync', 'start_datetime': 'tomorrow 10am'});
      expect(d.action, _insert);
      expect(d.data, 'content://com.android.calendar/events');
      expect(d.arguments['title'], 'Team sync');
      expect(d.arguments['beginTime'], DateTime(2026, 9, 13, 10, 0).millisecondsSinceEpoch);
      expect(d.arguments['endTime'], DateTime(2026, 9, 13, 11, 0).millisecondsSinceEpoch);
      expect(d.arguments.containsKey('allDay'), isFalse);
    });

    test('date-only calendar event is all-day', () {
      final d = intent('create_calendar_event',
          {'title': 'Trip', 'start_datetime': '2026-09-20'});
      expect(d.arguments['allDay'], true);
      expect(d.arguments['beginTime'], DateTime(2026, 9, 20).millisecondsSinceEpoch);
    });

    test('reminder with a time is a timed event today', () {
      final d = intent('set_reminder', {'text': 'call dentist', 'time': '15:00'});
      expect(d.action, _insert);
      expect(d.arguments['title'], 'call dentist');
      expect(d.arguments['beginTime'], DateTime(2026, 9, 12, 15, 0).millisecondsSinceEpoch);
    });

    test('reminder without a time is all-day today', () {
      final d = intent('set_reminder', {'text': 'buy milk'});
      expect(d.arguments['allDay'], true);
      expect(d.arguments['beginTime'], DateTime(2026, 9, 12).millisecondsSinceEpoch);
    });
  });

  group('open_settings', () {
    test('known panel', () {
      expect(intent('open_settings', {'panel': 'wifi'}).action,
          'android.settings.WIFI_SETTINGS');
      expect(intent('open_settings', {'panel': 'bluetooth'}).action,
          'android.settings.BLUETOOTH_SETTINGS');
    });

    test('unknown panel is unhandled', () {
      final d = buildDispatch(ToolCall('open_settings', {'panel': 'nfc'}), now: now);
      expect(d, isA<UnknownDispatch>());
    });
  });

  group('drafts', () {
    test('sms uses smsto and a body extra', () {
      final d = intent('draft_sms', {'recipient': 'Mom', 'body': 'On my way'});
      expect(d.action, _sendto);
      expect(d.data, 'smsto:Mom');
      expect(d.arguments['sms_body'], 'On my way');
    });

    test('email encodes subject and body into the mailto uri', () {
      final d = intent('draft_email',
          {'to': 'a@b.com', 'subject': 'Hi there', 'body': 'Lunch tomorrow?'});
      expect(d.action, _sendto);
      expect(d.data, startsWith('mailto:a%40b.com?'));
      expect(d.data, contains('subject=Hi%20there'));
      expect(d.data, contains('body=Lunch%20tomorrow%3F'));
      expect(d.arguments['android.intent.extra.SUBJECT'], 'Hi there');
    });
  });

  group('play_music', () {
    test('query plays from search', () {
      final d = intent('play_music', {'query': 'lo-fi beats'});
      expect(d.action, _playFromSearch);
      expect(d.arguments['query'], 'lo-fi beats');
    });

    test('transport action goes native', () {
      expect(native('play_music', {'action': 'pause'}).args['action'], 'pause');
    });

    test('unknown/empty action defaults to play', () {
      expect(native('play_music', {'action': 'garbage'}).args['action'], 'play');
      expect(native('play_music', {}).args['action'], 'play');
    });
  });

  group('native tools', () {
    test('flashlight on/off', () {
      expect(native('toggle_flashlight', {'state': 'on'}).method, 'setTorch');
      expect(native('toggle_flashlight', {'state': 'on'}).args['on'], true);
      expect(native('toggle_flashlight', {'state': 'off'}).args['on'], false);
    });
  });

  group('create_note', () {
    test('send text/plain with the content', () {
      final d = intent('create_note', {'content': 'buy milk', 'title': 'Todo'});
      expect(d.action, _send);
      expect(d.type, 'text/plain');
      expect(d.arguments['android.intent.extra.TEXT'], 'buy milk');
      expect(d.arguments['android.intent.extra.SUBJECT'], 'Todo');
    });
  });

  test('unknown tool is unhandled', () {
    expect(buildDispatch(ToolCall('teleport', {}), now: now), isA<UnknownDispatch>());
  });
}
