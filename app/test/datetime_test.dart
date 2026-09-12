import 'package:flutter_test/flutter_test.dart';

import 'package:offhand/agent/datetime.dart';

void main() {
  group('parseClockTime', () {
    test('24-hour HH:MM', () {
      expect(parseClockTime('07:00'), const ClockTime(7, 0));
      expect(parseClockTime('18:30'), const ClockTime(18, 30));
    });

    test('12-hour with meridiem', () {
      expect(parseClockTime('7am'), const ClockTime(7, 0));
      expect(parseClockTime('7:30 pm'), const ClockTime(19, 30));
      expect(parseClockTime('12am'), const ClockTime(0, 0));
      expect(parseClockTime('12pm'), const ClockTime(12, 0));
    });

    test('words', () {
      expect(parseClockTime('noon'), const ClockTime(12, 0));
      expect(parseClockTime('midnight'), const ClockTime(0, 0));
    });

    test('rejects nonsense and out-of-range', () {
      expect(parseClockTime('later'), isNull);
      expect(parseClockTime('25:00'), isNull);
      expect(parseClockTime('10:99'), isNull);
    });
  });

  group('parseDateTime', () {
    final now = DateTime(2026, 9, 12, 8, 0);

    test('ISO with time has a time', () {
      final p = parseDateTime('2026-09-20T14:30:00', now: now)!;
      expect(p.hasTime, isTrue);
      expect(p.dateTime, DateTime(2026, 9, 20, 14, 30));
    });

    test('ISO date only has no time', () {
      final p = parseDateTime('2026-09-20', now: now)!;
      expect(p.hasTime, isFalse);
      expect(p.dateTime, DateTime(2026, 9, 20));
    });

    test('tomorrow with a time', () {
      final p = parseDateTime('tomorrow 10am', now: now)!;
      expect(p.hasTime, isTrue);
      expect(p.dateTime, DateTime(2026, 9, 13, 10, 0));
    });

    test('bare time is today', () {
      final p = parseDateTime('15:00', now: now)!;
      expect(p.hasTime, isTrue);
      expect(p.dateTime, DateTime(2026, 9, 12, 15, 0));
    });

    test('day word without a time', () {
      final p = parseDateTime('tomorrow', now: now)!;
      expect(p.hasTime, isFalse);
      expect(p.dateTime, DateTime(2026, 9, 13));
    });

    test('unparseable returns null', () {
      expect(parseDateTime('whenever', now: now), isNull);
    });
  });
}
