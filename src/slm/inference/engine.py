"""In-process generation with full sampling control.

Exactly one model is resident at a time. GPU training jobs call `engine.block()` to evict
it and keep it evicted until the job finishes, since training and inference can't share
16 GB.

All MLX work runs on one dedicated thread. MLX keeps thread-local state (including a cache of
compiled sampling functions that hold Python objects); if that state lives on the main thread
it is destroyed after the interpreter finalizes and segfaults the process on exit. An executor
thread is torn down while Python is still alive, and one thread also keeps MLX usage serial.
"""

import gc
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.95, ge=0, le=1)
    top_k: int = Field(0, ge=0)  # 0 = off
    min_p: float = Field(0.0, ge=0, le=1)
    repetition_penalty: float | None = Field(1.05, ge=0.5, le=2)
    repetition_context_size: int = Field(64, ge=1)
    presence_penalty: float | None = Field(None, ge=-2, le=2)
    frequency_penalty: float | None = Field(None, ge=-2, le=2)
    xtc_probability: float = Field(0.0, ge=0, le=1)
    xtc_threshold: float = Field(0.0, ge=0, le=0.5)
    max_tokens: int = Field(512, ge=1, le=8192)
    seed: int | None = None
    chat_template: str | None = None  # override the tokenizer's template
    enable_thinking: bool | None = None  # for templates that support it (e.g. Qwen3)


class EngineBusy(RuntimeError):
    pass


_DONE = object()


class Engine:
    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._model = None
        self._tokenizer = None
        self._key: tuple[str, str | None] | None = None
        self._blocked_reason: str | None = None

    def _on_mlx_thread(self, fn: Callable[[], Any]) -> Any:
        return self._pool.submit(fn).result()

    @property
    def loaded(self) -> dict | None:
        if self._key is None:
            return None
        return {"model_path": self._key[0], "adapter_path": self._key[1]}

    @property
    def blocked(self) -> str | None:
        return self._blocked_reason

    def block(self, reason: str) -> None:
        """Evict the model; waits for any in-flight generation to finish first."""
        self._blocked_reason = reason
        self._on_mlx_thread(self._unload)

    def unblock(self) -> None:
        self._blocked_reason = None

    def unload(self) -> None:
        self._on_mlx_thread(self._unload)

    def shutdown(self) -> None:
        """Tear the MLX thread down while the interpreter is still alive (see the module docstring)."""
        if self._key is not None:
            self._on_mlx_thread(self._unload)
        self._pool.shutdown(wait=True)

    def _unload(self) -> None:
        import mlx.core as mx

        self._model = self._tokenizer = self._key = None
        gc.collect()
        mx.clear_cache()

    def _ensure(self, model_path: str, adapter_path: str | None = None):
        from mlx_lm import load

        if self._blocked_reason:
            raise EngineBusy(self._blocked_reason)
        key = (model_path, adapter_path)
        if self._key != key:
            self._unload()
            self._model, self._tokenizer = load(model_path, adapter_path=adapter_path)
            self._key = key
        return self._model, self._tokenizer

    def preload(self, model_path: str, adapter_path: str | None = None) -> None:
        self._on_mlx_thread(lambda: self._ensure(model_path, adapter_path))

    def _prompt(self, tokenizer, messages: list[dict], params: SamplingParams):
        kwargs: dict = {"add_generation_prompt": True, "tokenize": True}
        if params.chat_template:
            kwargs["chat_template"] = params.chat_template
        if params.enable_thinking is not None:
            kwargs["enable_thinking"] = params.enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)

    def stream(
        self,
        messages: list[dict],
        params: SamplingParams,
        *,
        model_path: str,
        adapter_path: str | None = None,
    ) -> Iterator[dict]:
        """Yield {"type": "token", "text"} events, then one {"type": "done", ...stats}.

        Generation runs on the MLX thread and hands events over through a queue. Closing this
        generator early (e.g. a client disconnect) stops generation at the next token.
        """
        events: queue.Queue = queue.Queue()
        stop = threading.Event()

        def produce() -> None:
            try:
                for ev in self._generate(messages, params, model_path, adapter_path, stop):
                    events.put(ev)
            except BaseException as e:  # re-raised on the consumer side
                events.put(e)
            finally:
                events.put(_DONE)

        self._pool.submit(produce)
        try:
            while (ev := events.get()) is not _DONE:
                if isinstance(ev, BaseException):
                    raise ev
                yield ev
        finally:
            stop.set()

    def _generate(
        self,
        messages: list[dict],
        params: SamplingParams,
        model_path: str,
        adapter_path: str | None,
        stop: threading.Event,
    ) -> Iterator[dict]:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        model, tokenizer = self._ensure(model_path, adapter_path)
        if params.seed is not None:
            mx.random.seed(params.seed)
        special = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        sampler = make_sampler(
            temp=params.temperature,
            top_p=params.top_p if params.top_p < 1 else 0.0,
            min_p=params.min_p,
            top_k=params.top_k,
            xtc_probability=params.xtc_probability,
            xtc_threshold=params.xtc_threshold,
            xtc_special_tokens=special,
        )
        processors = make_logits_processors(
            repetition_penalty=params.repetition_penalty,
            repetition_context_size=params.repetition_context_size,
            presence_penalty=params.presence_penalty,
            frequency_penalty=params.frequency_penalty,
        )
        prompt = self._prompt(tokenizer, messages, params)
        start = time.perf_counter()
        last = None
        for resp in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=params.max_tokens,
            sampler=sampler,
            logits_processors=processors,
        ):
            last = resp
            if resp.text:
                yield {"type": "token", "text": resp.text}
            if stop.is_set():
                break
        yield {
            "type": "done",
            "prompt_tokens": last.prompt_tokens if last else 0,
            "generation_tokens": last.generation_tokens if last else 0,
            "tokens_per_sec": round(last.generation_tps, 1) if last else 0,
            "peak_mem_gb": round(last.peak_memory, 2) if last else 0,
            "finish_reason": "stopped" if stop.is_set() else (last.finish_reason if last else None),
            "seconds": round(time.perf_counter() - start, 2),
        }

    def generate(self, messages: list[dict], params: SamplingParams, **where) -> tuple[str, dict]:
        text, stats = [], {}
        for ev in self.stream(messages, params, **where):
            if ev["type"] == "token":
                text.append(ev["text"])
            else:
                stats = ev
        return "".join(text), stats


engine = Engine()
