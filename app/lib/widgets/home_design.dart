import 'dart:math' as math;
import 'package:flutter/material.dart';

const offhandOrange = Color(0xFFD95532);
const offhandInk = Color(0xFF29252D);

/// The same small sunburst is used in the wordmark and the voice control.
class OffhandMark extends StatelessWidget {
  const OffhandMark({super.key, this.size = 28, this.color = offhandOrange});
  final double size;
  final Color color;

  @override
  Widget build(BuildContext context) => ExcludeSemantics(
    child: CustomPaint(size: Size.square(size), painter: _MarkPainter(color)),
  );
}

class _MarkPainter extends CustomPainter {
  const _MarkPainter(this.color);
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.translate(size.width / 2, size.height / 2);
    final paint = Paint()..color = color;
    for (var i = 0; i < 10; i++) {
      canvas.save();
      canvas.rotate(i * math.pi / 5);
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(
            -size.width * .065,
            -size.height * .48,
            size.width * .13,
            size.height * .34,
          ),
          Radius.circular(size.width * .065),
        ),
        paint,
      );
      canvas.restore();
    }
    canvas.drawCircle(Offset.zero, size.width * .105, paint);
  }

  @override
  bool shouldRepaint(_MarkPainter oldDelegate) => oldDelegate.color != color;
}

class VoiceCard extends StatelessWidget {
  const VoiceCard({
    super.key,
    required this.loaded,
    required this.loading,
    required this.listening,
    required this.onStart,
    required this.onSpeak,
  });
  final bool loaded;
  final bool loading;
  final bool listening;
  final VoidCallback? onStart;
  final VoidCallback? onSpeak;

  @override
  Widget build(BuildContext context) {
    final largeText = MediaQuery.textScalerOf(context).scale(14) > 20;
    final copy = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          listening ? 'GO AHEAD, I’M LISTENING' : 'A LITTLE HELP, RIGHT HERE',
          style: const TextStyle(
            color: Color(0xFFC5B9C9),
            fontSize: 9,
            letterSpacing: 1.6,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 12),
        Text(
          listening ? 'What’s on\nyour mind?' : 'One less thing\non your mind.',
          style: const TextStyle(
            fontFamily: 'InstrumentSerif',
            color: Color(0xFFFFF7ED),
            fontSize: 32,
            height: 1.05,
          ),
        ),
        const SizedBox(height: 16),
        if (!loaded)
          TextButton(
            style: TextButton.styleFrom(
              foregroundColor: const Color(0xFFFFF7ED),
              backgroundColor: Colors.white.withValues(alpha: .10),
              disabledForegroundColor: const Color(0xFFC5B9C9),
              padding: const EdgeInsets.symmetric(horizontal: 14),
              minimumSize: const Size(48, 48),
              shape: const StadiumBorder(),
            ),
            onPressed: onStart,
            child: Text(
              loading ? 'Starting…' : 'Start assistant',
              style: const TextStyle(fontSize: 12),
            ),
          )
        else
          Text(
            listening ? 'Tap again to finish' : 'Tap the orb to speak',
            style: const TextStyle(
              color: Color(0xFFD8CCD8),
              fontSize: 11,
              height: 1.5,
            ),
          ),
      ],
    );
    final orb = SizedBox(
      width: 132,
      height: 152,
      child: CustomPaint(
        painter: _OrbitPainter(listening: listening),
        child: Center(
          child: Tooltip(
            message: loaded
                ? (listening ? 'Stop listening' : 'Speak your request')
                : 'Start assistant',
            child: Semantics(
              button: true,
              label: loaded
                  ? (listening ? 'Stop listening' : 'Speak your request')
                  : 'Start assistant',
              child: Material(
                color: Colors.transparent,
                shape: const CircleBorder(),
                clipBehavior: Clip.antiAlias,
                child: Ink(
                  width: 84,
                  height: 84,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      center: const Alignment(-.45, -.6),
                      radius: 1.2,
                      colors: listening
                          ? const [
                              Color(0xFFFFDDAA),
                              Color(0xFFF09968),
                              Color(0xFFCC623C),
                            ]
                          : const [
                              Color(0xFFFFCAA0),
                              Color(0xFFF58A5E),
                              Color(0xFFD95734),
                            ],
                    ),
                    border: Border.all(
                      color: const Color(0xFFFFCDAA).withValues(alpha: .6),
                    ),
                  ),
                  child: InkWell(
                    customBorder: const CircleBorder(),
                    onTap: loaded ? onSpeak : onStart,
                    child: Center(
                      child: loading
                          ? const SizedBox(
                              width: 22,
                              height: 22,
                              child: CircularProgressIndicator(
                                color: offhandInk,
                                strokeWidth: 2,
                              ),
                            )
                          : Icon(
                              listening
                                  ? Icons.stop_rounded
                                  : Icons.graphic_eq_rounded,
                              color: const Color(0xFF6C2E22),
                              size: 32,
                            ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
    return Container(
      padding: const EdgeInsets.fromLTRB(22, 22, 12, 22),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF302C38), Color(0xFF433244)],
        ),
        boxShadow: [
          BoxShadow(
            color: offhandInk.withValues(alpha: .12),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: largeText
          ? Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                copy,
                Center(child: orb),
              ],
            )
          : Row(
              children: [
                Expanded(child: copy),
                orb,
              ],
            ),
    );
  }
}

class _OrbitPainter extends CustomPainter {
  const _OrbitPainter({required this.listening});
  final bool listening;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    canvas.drawCircle(
      center,
      66,
      Paint()
        ..shader = RadialGradient(
          colors: [
            offhandOrange.withValues(alpha: .23),
            offhandOrange.withValues(alpha: 0),
          ],
        ).createShader(Rect.fromCircle(center: center, radius: 66)),
    );
    final line = Paint()
      ..color = const Color(0xFFD7BACF).withValues(alpha: .22)
      ..style = PaintingStyle.stroke
      ..strokeWidth = .7;
    canvas.save();
    canvas.translate(center.dx, center.dy);
    canvas.rotate(-.38);
    canvas.drawOval(
      Rect.fromCenter(center: Offset.zero, width: 118, height: 140),
      line,
    );
    canvas.rotate(.85);
    canvas.drawOval(
      Rect.fromCenter(center: Offset.zero, width: 116, height: 140),
      line,
    );
    canvas.restore();
    canvas.drawCircle(
      center.translate(49, -42),
      3,
      Paint()..color = const Color(0xFFFDCDAA),
    );
    canvas.drawCircle(
      center.translate(-49, 42),
      2,
      Paint()..color = const Color(0xFFB39DAF),
    );
  }

  @override
  bool shouldRepaint(_OrbitPainter oldDelegate) =>
      listening != oldDelegate.listening;
}

class SuggestionCard extends StatelessWidget {
  const SuggestionCard({
    super.key,
    required this.label,
    required this.detail,
    required this.icon,
    required this.tint,
    required this.onTap,
  });
  final String label;
  final String detail;
  final IconData icon;
  final Color tint;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final dark = Theme.of(context).brightness == Brightness.dark;
    final foreground = dark ? const Color(0xFFF5EDE4) : offhandInk;
    return Material(
      color: dark
          ? Color.alphaBlend(
              tint.withValues(alpha: .13),
              const Color(0xFF27252B),
            )
          : tint,
      borderRadius: BorderRadius.circular(22),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Opacity(
            opacity: onTap == null ? .45 : 1,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 38,
                      height: 38,
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: .45),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(icon, size: 21, color: offhandInk),
                    ),
                    const Spacer(),
                    Icon(
                      Icons.north_east_rounded,
                      size: 16,
                      color: foreground.withValues(alpha: .5),
                    ),
                  ],
                ),
                const SizedBox(height: 18),
                Text(
                  label,
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: foreground,
                    letterSpacing: -.3,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  detail,
                  style: TextStyle(
                    fontSize: 11,
                    color: foreground.withValues(alpha: .68),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
