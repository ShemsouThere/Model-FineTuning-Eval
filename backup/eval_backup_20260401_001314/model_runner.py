from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class UnsupportedEndpointError(RuntimeError):
    """Raised when the target llama.cpp build does not expose a requested API."""


@dataclass
class ModelConfig:
    name: str
    model_path: Path
    ctx_size: int = 8192
    threads: int = 4
    batch_size: int = 512
    ubatch_size: int = 512
    gpu_layers: int = 0
    extra_args: list[str] = field(default_factory=list)

    @property
    def safe_name(self) -> str:
        cleaned = [
            character.lower() if character.isalnum() else "_"
            for character in self.name.strip()
        ]
        name = "".join(cleaned).strip("_")
        while "__" in name:
            name = name.replace("__", "_")
        return name or "model"


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 256
    repeat_penalty: float = 1.05
    seed: int = 0
    stop: list[str] = field(default_factory=lambda: ["```"])
    cache_prompt: bool = True
    timeout_sec: int = 600


@dataclass
class GenerationResult:
    text: str
    latency_sec: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_response: dict[str, Any] | None = None


class LlamaCppServer:
    def __init__(
        self,
        binary_path: str | Path,
        model_config: ModelConfig,
        host: str,
        port: int,
        log_dir: Path,
        startup_timeout_sec: int = 900,
    ) -> None:
        self.binary_path = str(binary_path)
        self.model_config = model_config
        self.host = host
        self.port = port
        self.log_dir = log_dir
        self.startup_timeout_sec = startup_timeout_sec
        self.base_url = f"http://{self.host}:{self.port}"
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._api_mode: str = "chat"

    def __enter__(self) -> "LlamaCppServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._process and self._process.poll() is None:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        server_log_path = self.log_dir / f"{self.model_config.safe_name}.server.log"
        self._log_handle = server_log_path.open("a", encoding="utf-8")

        command = [
            self.binary_path,
            "-m",
            str(self.model_config.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.model_config.ctx_size),
            "-t",
            str(self.model_config.threads),
            "-b",
            str(self.model_config.batch_size),
            "-ub",
            str(self.model_config.ubatch_size),
            "-ngl",
            str(self.model_config.gpu_layers),
        ] + list(self.model_config.extra_args)

        LOGGER.info("Starting llama.cpp server for %s", self.model_config.name)
        LOGGER.debug("llama-server command: %s", command)
        self._process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.model_config.model_path.parent),
        )
        self._wait_until_ready()

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            LOGGER.info("Stopping llama.cpp server for %s", self.model_config.name)
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=10)
        self._process = None

        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def generate(
        self,
        messages: list[dict[str, str]],
        sampling: SamplingConfig,
    ) -> GenerationResult:
        if self._api_mode == "chat":
            try:
                return self._generate_chat(messages, sampling)
            except UnsupportedEndpointError:
                LOGGER.warning(
                    "OpenAI-compatible chat endpoint unavailable for %s; falling back to /completion",
                    self.model_config.name,
                )
                self._api_mode = "completion"

        prompt = self._render_fallback_prompt(messages)
        return self._generate_completion(prompt, sampling)

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_sec
        last_error = "server did not report readiness"
        while time.time() < deadline:
            if self._process is None:
                raise RuntimeError("llama.cpp server process was not created")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp server exited early for {self.model_config.name}. "
                    f"Check {self.log_dir / f'{self.model_config.safe_name}.server.log'}"
                )
            try:
                if self._healthcheck():
                    return
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
            time.sleep(1.0)
        raise TimeoutError(
            f"Timed out waiting for llama.cpp server for {self.model_config.name}: {last_error}"
        )

    def _healthcheck(self) -> bool:
        try:
            response = urllib.request.urlopen(
                f"{self.base_url}/health",
                timeout=5,
            )
            return response.status == 200
        except Exception:
            response = urllib.request.urlopen(
                f"{self.base_url}/v1/models",
                timeout=5,
            )
            return response.status == 200

    def _generate_chat(
        self,
        messages: list[dict[str, str]],
        sampling: SamplingConfig,
    ) -> GenerationResult:
        payload = {
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": sampling.seed,
            "stream": False,
            "stop": sampling.stop,
        }
        started = time.time()
        try:
            response = self._post_json(
                f"{self.base_url}/v1/chat/completions",
                payload,
                timeout=sampling.timeout_sec,
            )
        except urllib.error.HTTPError as exc:
            if exc.code in {400, 404, 405, 422, 501}:
                raise UnsupportedEndpointError(str(exc)) from exc
            raise

        latency = time.time() - started
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})
        return GenerationResult(
            text=str(message.get("content", "")).strip(),
            latency_sec=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response=response,
        )

    def _generate_completion(
        self,
        prompt: str,
        sampling: SamplingConfig,
    ) -> GenerationResult:
        payload = {
            "prompt": prompt,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "n_predict": sampling.max_tokens,
            "repeat_penalty": sampling.repeat_penalty,
            "seed": sampling.seed,
            "stop": sampling.stop,
            "cache_prompt": sampling.cache_prompt,
            "stream": False,
        }
        started = time.time()
        response = self._post_json(
            f"{self.base_url}/completion",
            payload,
            timeout=sampling.timeout_sec,
        )
        latency = time.time() - started

        text = response.get("content")
        if text is None:
            choices = response.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
        usage = response.get("usage", {})
        return GenerationResult(
            text=str(text or "").strip(),
            latency_sec=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            raw_response=response,
        )

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _render_fallback_prompt(messages: list[dict[str, str]]) -> str:
        blocks: list[str] = []
        for message in messages:
            role = message.get("role", "user").strip().upper()
            content = message.get("content", "").strip()
            blocks.append(f"{role}:\n{content}")
        blocks.append("ASSISTANT:")
        return "\n\n".join(blocks)
