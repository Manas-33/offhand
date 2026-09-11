package com.offhand.offhand

/**
 * A generation backend. The rest of the app talks only to this interface, so the
 * mock and the real on-device runtime are interchangeable.
 *
 * [generate] runs asynchronously and reports progress through [emit], which is
 * called with event maps of the form:
 *   {"type": "token",  "text": String}
 *   {"type": "done",   "ttftMs": Long, "tps": Double, "tokens": Int, "totalMs": Long, "cancelled": Boolean}
 *   {"type": "error",  "message": String}
 * emit may be called from any thread; MainActivity forwards to the Flutter side.
 */
interface Engine {
    fun loadModel(modelPath: String): Boolean
    fun generate(prompt: String, emit: (Map<String, Any?>) -> Unit)
    fun stop()
}
