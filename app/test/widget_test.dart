import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offhand/agent/parser.dart';
import 'package:offhand/main.dart';
import 'package:offhand/widgets/approval_sheet.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const codec = StandardMethodCodec();
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  final controls = <MethodCall>[];
  final actions = <MethodCall>[];
  var failLoad = false;

  setUp(() {
    controls.clear();
    actions.clear();
    failLoad = false;
    messenger.setMockMethodCallHandler(
      const MethodChannel('com.offhand/tokens'),
      (_) async => null,
    );
    messenger.setMockMethodCallHandler(
      const MethodChannel('com.offhand/control'),
      (call) async {
        controls.add(call);
        if (call.method == 'loadModel') {
          if (failLoad) throw PlatformException(code: 'load_failed');
          return true;
        }
        return null;
      },
    );
    messenger.setMockMethodCallHandler(
      const MethodChannel('com.offhand/tools'),
      (call) async {
        actions.add(call);
        return true;
      },
    );
  });

  Future<void> emit(WidgetTester tester, Map<String, Object> event) async {
    await messenger.handlePlatformMessage(
      'com.offhand/tokens',
      codec.encodeSuccessEnvelope(event),
      (_) {},
    );
    await tester.pump();
  }

  Future<void> tap(WidgetTester tester, Finder finder) async {
    await tester.ensureVisible(finder);
    await tester.tap(finder);
    await tester.pumpAndSettle();
  }

  Future<void> startRequest(WidgetTester tester) async {
    await tester.pumpWidget(const OffhandApp());
    await tap(tester, find.text('Start assistant'));
    await tester.enterText(find.byType(TextField), 'Turn on the flashlight');
    await tester.pump();
    await tester.ensureVisible(find.byTooltip('Ask Offhand'));
    await tester.tap(find.byTooltip('Ask Offhand'));
    await tester.pump();
    await emit(tester, {
      'type': 'token',
      'text':
          '<tool_call>{"name":"toggle_flashlight","arguments":{"state":"on"}}</tool_call>',
    });
  }

  Map<String, Object> done({bool cancelled = false}) => {
    'type': 'done',
    'ttftMs': 100,
    'tps': 20.0,
    'tokens': 12,
    'totalMs': 600,
    'cancelled': cancelled,
  };

  testWidgets(
    'examples fill the request without executing and empty requests stay disabled',
    (tester) async {
      await tester.pumpWidget(const OffhandApp());
      await tap(tester, find.text('Start assistant'));
      expect(
        tester
            .widget<IconButton>(
              find.byWidgetPredicate(
                (widget) =>
                    widget is IconButton && widget.tooltip == 'Ask Offhand',
              ),
            )
            .onPressed,
        isNull,
      );
      await tap(tester, find.text('10-minute timer'));
      expect(
        tester.widget<TextField>(find.byType(TextField)).controller!.text,
        'Start a 10 minute timer',
      );
      expect(controls.where((call) => call.method == 'generate'), isEmpty);
      await tester.enterText(find.byType(TextField), '   ');
      await tester.pump();
      expect(
        tester
            .widget<IconButton>(
              find.byWidgetPredicate(
                (widget) =>
                    widget is IconButton && widget.tooltip == 'Ask Offhand',
              ),
            )
            .onPressed,
        isNull,
      );
    },
  );

  testWidgets('setup failure explains recovery and allows retry', (
    tester,
  ) async {
    failLoad = true;
    await tester.pumpWidget(const OffhandApp());
    await tap(tester, find.text('Start assistant'));
    expect(
      find.textContaining('Couldn’t start the assistant.'),
      findsOneWidget,
    );
    failLoad = false;
    await tap(tester, find.text('Start assistant'));
    expect(find.text('On-device assistant · Ready'), findsOneWidget);
    expect(find.textContaining('Couldn’t start the assistant.'), findsNothing);
  });

  testWidgets(
    'cancelled generation never opens approval or executes an action',
    (tester) async {
      await startRequest(tester);
      await emit(tester, done(cancelled: true));
      await tester.pumpAndSettle();
      expect(
        find.text('Request stopped. No action was taken.'),
        findsOneWidget,
      );
      expect(find.text('Approve action'), findsNothing);
      expect(actions, isEmpty);
    },
  );

  testWidgets(
    'stop suppresses approval even if completion races cancellation',
    (tester) async {
      await startRequest(tester);
      await tester.ensureVisible(find.byTooltip('Stop request'));
      await tester.tap(find.byTooltip('Stop request'));
      await tester.pump();
      expect(controls.last.method, 'stop');
      await emit(tester, done());
      await tester.pumpAndSettle();
      expect(
        find.text('Request stopped. No action was taken.'),
        findsOneWidget,
      );
      expect(find.text('Approve action'), findsNothing);
      expect(actions, isEmpty);
    },
  );

  testWidgets('action only executes after explicit approval', (tester) async {
    await startRequest(tester);
    await emit(tester, done());
    await tester.pumpAndSettle();
    expect(find.text('Turn the flashlight on'), findsOneWidget);
    expect(actions, isEmpty);
    await tap(tester, find.text('Not now'));
    expect(actions, isEmpty);
    expect(find.text('Not approved. No action was taken.'), findsOneWidget);
    await tap(tester, find.text('Review action'));
    expect(actions, isEmpty);
    await tap(tester, find.text('Approve action'));
    expect(actions.single.method, 'setTorch');
    expect(find.text('Flashlight on'), findsOneWidget);
  });

  testWidgets('home scrolls at large text sizes with keyboard and dark mode', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(320, 640);
    tester.view.devicePixelRatio = 1;
    tester.view.viewInsets = const FakeViewPadding(bottom: 240);
    tester.platformDispatcher.textScaleFactorTestValue = 2;
    tester.platformDispatcher.platformBrightnessTestValue = Brightness.dark;
    addTearDown(tester.view.reset);
    addTearDown(tester.platformDispatcher.clearAllTestValues);
    await tester.pumpWidget(const OffhandApp());
    await tester.pumpAndSettle();
    await tap(tester, find.text('Start assistant'));
    await tester.ensureVisible(find.byTooltip('Ask Offhand'));
    expect(tester.takeException(), isNull);
  });

  testWidgets('long draft review scrolls to approval on a small screen', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(320, 640);
    tester.view.devicePixelRatio = 1;
    tester.platformDispatcher.textScaleFactorTestValue = 2;
    addTearDown(tester.view.reset);
    addTearDown(tester.platformDispatcher.clearAllTestValues);
    bool? approved;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Builder(
            builder: (context) {
              return TextButton(
                onPressed: () async {
                  approved = await showApprovalSheet(
                    context,
                    ToolCall('draft_sms', {
                      'recipient': 'Alex',
                      'body': List.filled(
                        20,
                        'A message to review.',
                      ).join('\n'),
                    }),
                  );
                },
                child: const Text('Review'),
              );
            },
          ),
        ),
      ),
    );
    await tap(tester, find.text('Review'));
    await tap(tester, find.text('Open draft'));
    expect(approved, isTrue);
    expect(tester.takeException(), isNull);
  });
}
