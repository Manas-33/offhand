package com.offhand.offhand

import kotlin.concurrent.thread

/**
 * Placeholder engine that streams a canned response token by token with
 * realistic timing, so the Flutter <-> platform-channel plumbing and the UI can
 * be verified end to end before the real NPU runtime is wired in via
 * [GenieEngine]. Swap MockEngine for GenieEngine in MainActivity when ready.
 */
class MockEngine : Engine {
    @Volatile private var cancelled = false
    private var loaded = false

    override fun loadModel(modelPath: String): Boolean {
        // The real engine maps the exported QAIRT context binary here.
        loaded = true
        return true
    }

    override fun generate(prompt: String, emit: (Map<String, Any?>) -> Unit) {
        cancelled = false
        thread(name = "mock-generate") {
            val tokens = mockReplyFor(prompt).split(" ")
            val start = System.currentTimeMillis()
            var ttft = -1L
            var count = 0
            for (word in tokens) {
                if (cancelled) break
                Thread.sleep(45) // ~22 tok/s, matching the 8 Elite decode rate from the device proof
                if (ttft < 0) ttft = System.currentTimeMillis() - start
                emit(mapOf("type" to "token", "text" to if (count == 0) word else " $word"))
                count++
            }
            val total = System.currentTimeMillis() - start
            emit(
                mapOf(
                    "type" to "done",
                    "ttftMs" to ttft.coerceAtLeast(0),
                    "tps" to if (total > 0) count * 1000.0 / total else 0.0,
                    "tokens" to count,
                    "totalMs" to total,
                    "cancelled" to cancelled,
                )
            )
        }
    }

    override fun stop() {
        cancelled = true
    }

    private fun mockReplyFor(prompt: String): String {
        val p = prompt.lowercase()
        // A no-call branch, so the irrelevance path is exercised too.
        if ("weather" in p || "capital" in p || "who " in p || "what is" in p) {
            return "I can't do that on-device, but I can set alarms, timers, reminders, " +
                "draft messages, and more."
        }
        val call = when {
            "alarm" in p -> """{"name": "set_alarm", "arguments": {"time": "07:00"}}"""
            "timer" in p -> """{"name": "set_timer", "arguments": {"duration_minutes": 10}}"""
            "remind" in p -> """{"name": "set_reminder", "arguments": {"text": "call the dentist", "time": "15:00"}}"""
            "calendar" in p || "meeting" in p || "appointment" in p || "schedule" in p || "event" in p ->
                """{"name": "create_calendar_event", "arguments": {"title": "Team sync", "start_datetime": "tomorrow 10am"}}"""
            "wifi" in p || "wi-fi" in p -> """{"name": "open_settings", "arguments": {"panel": "wifi"}}"""
            "bluetooth" in p -> """{"name": "open_settings", "arguments": {"panel": "bluetooth"}}"""
            "settings" in p || "display" in p || "brightness" in p -> """{"name": "open_settings", "arguments": {"panel": "display"}}"""
            "email" in p || "mail" in p -> """{"name": "draft_email", "arguments": {"to": "alex@example.com", "subject": "Lunch", "body": "Want to grab lunch tomorrow?"}}"""
            "text" in p || "message" in p || "sms" in p -> """{"name": "draft_sms", "arguments": {"recipient": "Mom", "body": "On my way!"}}"""
            "pause" in p -> """{"name": "play_music", "arguments": {"action": "pause"}}"""
            "next" in p || "skip" in p -> """{"name": "play_music", "arguments": {"action": "next"}}"""
            "play" in p || "music" in p || "song" in p -> """{"name": "play_music", "arguments": {"query": "lo-fi beats"}}"""
            "flashlight" in p || "torch" in p || "light" in p -> """{"name": "toggle_flashlight", "arguments": {"state": "on"}}"""
            "note" in p -> """{"name": "create_note", "arguments": {"content": "$prompt"}}"""
            else -> """{"name": "create_note", "arguments": {"content": "$prompt"}}"""
        }
        // Wrapped so the Dart parser (mirror of parse.py) can extract it.
        return "<tool_call> $call </tool_call>"
    }
}
