package com.offhand.offhand

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.TextView
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import org.json.JSONObject

/**
 * Debug-only batch runner for the on-device eval replay, started over adb by
 * eval/run_device_eval.py (declared in the debug manifest only).
 *
 * Sends every item in files/eval/items.jsonl through the same GenieEngine the
 * app uses, one fresh request at a time, and writes each raw output with the
 * runtime's own timing to files/eval/results.jsonl. Scoring happens on the
 * host with the eval harness, so this side only records what the model said.
 * results.jsonl appears only once every item has run (it is written as .part
 * and renamed); a failure writes files/eval/error.txt instead.
 */
class EvalActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Keep the screen on: a screen-off mid-run backgrounds the app.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val status = TextView(this).apply {
            textSize = 18f
            setPadding(48, 96, 48, 48)
        }
        setContentView(status)
        val show = { text: String -> runOnUiThread { status.text = text } }

        // onCreate runs again on a configuration change; never start a second run.
        if (!running.compareAndSet(false, true)) {
            show("Eval already running")
            return
        }
        val modelDir = intent.getStringExtra(EXTRA_MODEL) ?: DEFAULT_MODEL_DIR
        val dir = File(filesDir, "eval").apply { mkdirs() }
        thread(name = "offhand-eval") {
            try {
                runEval(dir, modelDir, show)
            } catch (e: Exception) {
                Log.e(TAG, "eval failed", e)
                File(dir, "error.txt").writeText(e.toString())
                show("Eval failed: ${e.message}")
            } finally {
                running.set(false)
            }
        }
    }

    private fun runEval(dir: File, modelDir: String, show: (String) -> Unit) {
        File(dir, "results.jsonl").delete()
        File(dir, "error.txt").delete()
        val items = File(dir, "items.jsonl").readLines().filter { it.isNotBlank() }.map { JSONObject(it) }

        show("Loading $modelDir")
        val engine = GenieEngine(applicationContext)
        check(engine.loadModel(modelDir)) { "model load failed: $modelDir" }

        val part = File(dir, "results.jsonl.part")
        part.bufferedWriter().use { out ->
            items.forEachIndexed { i, item ->
                val id = item.getString("id")
                show("Running ${i + 1}/${items.size}: $id")
                val row = runOne(engine, item.getString("query")).put("id", id)
                out.write(row.toString())
                out.newLine()
                out.flush()
                Log.i(TAG, "eval ${i + 1}/${items.size} $id")
            }
        }
        check(part.renameTo(File(dir, "results.jsonl"))) { "could not finalize ${part.name}" }
        Log.i(TAG, "eval done: ${items.size} items")
        show("Done: ${items.size} items")
    }

    /** One request through the app's own generate path; blocks until it finishes. */
    private fun runOne(engine: Engine, query: String): JSONObject {
        val raw = StringBuffer()
        var done: Map<String, Any?>? = null
        var error: String? = null
        val latch = CountDownLatch(1)
        engine.generate(query) { event ->
            when (event["type"]) {
                "token" -> raw.append(event["text"]?.toString().orEmpty())
                "done" -> {
                    done = event
                    latch.countDown()
                }
                "error" -> {
                    error = event["message"]?.toString() ?: "generation failed"
                    latch.countDown()
                }
            }
        }
        // A hung request would leave the runtime mid-decode, and the next one
        // would race it on the same handle, so stop the whole run instead.
        check(latch.await(REQUEST_TIMEOUT_S, TimeUnit.SECONDS)) {
            "timed out after ${REQUEST_TIMEOUT_S}s on: $query"
        }

        val row = JSONObject().put("query", query).put("raw", raw.toString())
        error?.let { row.put("error", it) }
        done?.let { d ->
            row.put("ttft_ms", json(d["ttftMs"]))
                .put("prompt_tokens", json(d["promptTokens"]))
                .put("generated_tokens", json(d["tokens"]))
                .put("prefill_tps", json(d["prefillTps"]))
                .put("decode_tps", json(d["tps"]))
                .put("stop_reason", json(d["stopReason"]))
        }
        return row
    }

    /** JSONObject rejects NaN and Infinity; record those, and nulls, as JSON null. */
    private fun json(value: Any?): Any = when {
        value == null -> JSONObject.NULL
        value is Double && !value.isFinite() -> JSONObject.NULL
        else -> value
    }

    companion object {
        private const val TAG = "OffhandEval"
        private const val EXTRA_MODEL = "model"

        /** Same bundle the app's screens load. */
        private const val DEFAULT_MODEL_DIR = "models/Qwen3-0.6B-Specialist"

        /** A normal request takes about a second; this only catches a hang. */
        private const val REQUEST_TIMEOUT_S = 120L

        private val running = AtomicBoolean(false)
    }
}
