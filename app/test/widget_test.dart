import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:offhand/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // The home screen subscribes to the token EventChannel on start; stub it so
  // the widget test does not touch a real platform channel.
  setUp(() {
    const channel = EventChannel('com.offhand/tokens');
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
      MethodChannel(channel.name, const StandardMethodCodec()),
      (MethodCall call) async => null,
    );
  });

  testWidgets('renders the core controls', (WidgetTester tester) async {
    await tester.pumpWidget(const OffhandApp());

    expect(find.text('Load model'), findsOneWidget);
    expect(find.text('Generate'), findsOneWidget);
    expect(find.text('Output will stream here.'), findsOneWidget);
  });
}
