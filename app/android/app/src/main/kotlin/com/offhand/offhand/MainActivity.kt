package com.offhand.offhand

import android.os.Handler
import android.os.Looper
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val controlChannel = "com.offhand/control"
    private val tokenChannel = "com.offhand/tokens"

    // Swap MockEngine() for GenieEngine() once the NPU runtime is wired in.
    private val engine: Engine = MockEngine()
    private var eventSink: EventChannel.EventSink? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger

        EventChannel(messenger, tokenChannel).setStreamHandler(
            object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, sink: EventChannel.EventSink?) {
                    eventSink = sink
                }

                override fun onCancel(arguments: Any?) {
                    eventSink = null
                }
            }
        )

        MethodChannel(messenger, controlChannel).setMethodCallHandler { call, result ->
            when (call.method) {
                "loadModel" -> {
                    val path = call.argument<String>("modelPath") ?: ""
                    result.success(engine.loadModel(path))
                }
                "generate" -> {
                    val prompt = call.argument<String>("prompt") ?: ""
                    engine.generate(prompt) { event ->
                        // EventSink must be touched on the main thread.
                        mainHandler.post { eventSink?.success(event) }
                    }
                    result.success(null)
                }
                "stop" -> {
                    engine.stop()
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }
    }
}
