/// Best-effort parsing of the loose time/date strings the model emits.
///
/// The tool schema says fields like `time` are "24-hour HH:MM" and
/// `start_datetime` is "ISO 8601 or natural form", but a small on-device model
/// will not always oblige. These parsers cover the common shapes (ISO-8601,
/// `HH:MM`, `7am`, `7:30 pm`, `noon`, and `today`/`tomorrow`/`tonight` with an
/// optional time) and return null when they cannot, so callers can fall back to
/// opening the relevant editor with the field left blank.
library;

/// A wall-clock time of day.
class ClockTime {
  const ClockTime(this.hour, this.minute);
  final int hour;
  final int minute;

  @override
  bool operator ==(Object other) =>
      other is ClockTime && other.hour == hour && other.minute == minute;

  @override
  int get hashCode => Object.hash(hour, minute);

  @override
  String toString() =>
      '${hour.toString().padLeft(2, '0')}:${minute.toString().padLeft(2, '0')}';
}

/// A parsed instant plus whether a time-of-day was actually present, so a caller
/// can decide between a timed event and an all-day one.
class ParsedDateTime {
  const ParsedDateTime(this.dateTime, {required this.hasTime});
  final DateTime dateTime;
  final bool hasTime;
}

final RegExp _hhmm = RegExp(r'^(\d{1,2}):(\d{2})\s*(am|pm)?$');
final RegExp _hAmPm = RegExp(r'^(\d{1,2})\s*(am|pm)$');
// A time token embedded anywhere in a longer phrase ("meeting at 3:30pm").
final RegExp _timeInText = RegExp(r'(\d{1,2}(?::\d{2})?)\s*(am|pm)|(\d{1,2}:\d{2})');
final RegExp _isoDateOnly = RegExp(r'^\d{4}-\d{2}-\d{2}$');

/// Parse a clock time such as `07:00`, `7am`, `7:30 pm`, `18:30`, `noon`.
ClockTime? parseClockTime(String raw) {
  final s = raw.trim().toLowerCase();
  if (s.isEmpty) return null;
  if (s == 'noon' || s == 'midday') return const ClockTime(12, 0);
  if (s == 'midnight') return const ClockTime(0, 0);

  var m = _hhmm.firstMatch(s);
  if (m != null) {
    return _apply(int.parse(m.group(1)!), int.parse(m.group(2)!), m.group(3));
  }
  m = _hAmPm.firstMatch(s);
  if (m != null) {
    return _apply(int.parse(m.group(1)!), 0, m.group(2));
  }
  return null;
}

ClockTime? _apply(int hour, int minute, String? meridiem) {
  var h = hour;
  if (meridiem == 'pm' && h < 12) h += 12;
  if (meridiem == 'am' && h == 12) h = 0;
  if (h < 0 || h > 23 || minute < 0 || minute > 59) return null;
  return ClockTime(h, minute);
}

/// Parse a date-time from ISO-8601 or a small set of natural forms.
///
/// [now] defaults to [DateTime.now]; pass it in tests for determinism.
ParsedDateTime? parseDateTime(String raw, {DateTime? now}) {
  final base = now ?? DateTime.now();
  final trimmed = raw.trim();
  if (trimmed.isEmpty) return null;

  // ISO first: a bare date has no time-of-day; anything else parseable does.
  if (_isoDateOnly.hasMatch(trimmed)) {
    final d = DateTime.tryParse(trimmed);
    if (d != null) return ParsedDateTime(d, hasTime: false);
  }
  final iso = DateTime.tryParse(trimmed);
  if (iso != null) return ParsedDateTime(iso, hasTime: true);

  final s = trimmed.toLowerCase();

  // Pick the day the phrase refers to.
  DateTime day = DateTime(base.year, base.month, base.day);
  if (s.contains('day after tomorrow')) {
    day = day.add(const Duration(days: 2));
  } else if (s.contains('tomorrow')) {
    day = day.add(const Duration(days: 1));
  } // "today" / "tonight" / bare time all stay on the current day.

  // Pull a time token out of the phrase, if any.
  final tm = _timeInText.firstMatch(s);
  if (tm != null) {
    final token = tm.group(0)!;
    final clock = parseClockTime(token);
    if (clock != null) {
      return ParsedDateTime(
        DateTime(day.year, day.month, day.day, clock.hour, clock.minute),
        hasTime: true,
      );
    }
  }

  // A day word with no time (e.g. "tomorrow").
  final hadDayWord = s.contains('today') ||
      s.contains('tonight') ||
      s.contains('tomorrow') ||
      s.contains('day after');
  if (hadDayWord) return ParsedDateTime(day, hasTime: false);

  return null;
}
