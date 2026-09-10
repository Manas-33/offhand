"""Model runners: turn a user query into raw model text.

The rest of the harness only depends on the ``ModelRunner`` protocol, so a new
backend (an AIMET QuantSim ONNX model, or the on-device Genie/llama.cpp bridge)
plugs in by implementing ``generate`` and nothing else changes.

``HFRunner`` (the FP16 / bnb-quantized reference) imports torch + transformers
lazily so that ``MockRunner`` and the scoring modules stay importable with no
heavy deps — that is what lets the test suite run on a laptop.
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_SYSTEM_PROMPT = (
    "You are Offhand, an on-device phone assistant. Use the provided tools to "
    "fulfill the user's request. Call exactly one tool when a request maps to a "
    "tool. If no tool applies, answer directly in plain text and do not call any "
    "tool."
)


class ModelRunner(Protocol):
    def generate(self, query: str) -> str: ...


class MockRunner:
    """Return canned outputs keyed by query. Used by the test suite.

    ``mapping`` maps a query string to the raw text the model would emit;
    unknown queries return ``default``.
    """

    def __init__(self, mapping: dict[str, str] | None = None, default: str = ""):
        self.mapping = mapping or {}
        self.default = default

    def generate(self, query: str) -> str:
        return self.mapping.get(query, self.default)


class HFRunner:
    """FP16 or bitsandbytes-quantized HuggingFace reference runner.

    ``quant`` is one of ``fp16`` | ``int8`` | ``nf4``. The bnb 4-bit / 8-bit
    paths are quick proxies for the M0c "does the curve exist?" spike; the real
    on-device W4A16 (AIMET QuantSim + SeqMSE/SpinQuant) checkpoints are evaluated
    in M2 through a separate runner that satisfies this same interface.
    """

    def __init__(
        self,
        model_id: str,
        tools: list[dict],
        quant: str = "fp16",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tools = tools
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if quant == "fp16":
            if device is not None:
                # Explicit device (mps/cpu/cuda) for local runs. fp16 is
                # unsupported for many CPU ops, so fall back to fp32 there.
                dtype = torch.float32 if device == "cpu" else torch.float16
                self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
                self._device = device
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype="auto", device_map="auto"
                )
                self._device = self.model.device
        elif quant in ("int8", "nf4"):
            if device not in (None, "cuda"):
                raise ValueError(
                    "int8/nf4 use bitsandbytes, which requires a CUDA GPU. "
                    "For a local (Mac) run use --config fp16."
                )
            from transformers import BitsAndBytesConfig

            if quant == "int8":
                qconfig = BitsAndBytesConfig(load_in_8bit=True)
            else:
                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=qconfig, torch_dtype="auto", device_map="auto"
            )
            self._device = self.model.device
        else:
            raise ValueError(f"unknown quant mode: {quant!r}")

    def generate(self, query: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
