"""Stdio MCP service for bounded, isolated C++ inference calls."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    text: str
    prompt_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=0)
    finish_reason: Literal["stop", "length"]
    generation_ms: float = Field(ge=0, allow_inf_nan=False)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    prompt_sha256: str
    engine_sha256: str
    model_dir: str
    max_tokens: int
    temperature: float
    seed: int
    raw: bool
    threads: int
    quantized: bool


def create_server(binary: Path, model_dir: Path, *, threads: int = 1,
                  quantize: bool = False, timeout: float = 120) -> FastMCP:
    binary, model_dir = binary.resolve(), model_dir.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"engine is not executable: {binary}; build mk-infer first")
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")
    if not 1 <= threads <= 64 or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("threads must be 1..64 and timeout must be finite and in (0, 300]")
    with binary.open("rb") as engine_file:
        engine_sha256 = hashlib.file_digest(engine_file, "sha256").hexdigest()
    server = FastMCP("vnni-llm")
    # ponytail: one subprocess at a time bounds model memory; reject excess work.
    lock = asyncio.Lock()

    @server.tool()
    async def generate(
        prompt: Annotated[str, Field(min_length=1, max_length=16384, strict=True)],
        max_tokens: Annotated[int, Field(ge=1, le=1024, strict=True)] = 128,
        temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0,
        raw: bool = False,
    ) -> GenerationResult:
        """Generate text. Prompt <=16384 UTF-8 bytes/2048 tokens; fixed seed 42.

        latency_ms includes process startup and weight loading. output_tokens
        counts emitted tokens and excludes EOS. Concurrent calls fail as busy.
        """
        encoded = prompt.encode("utf-8")
        if len(encoded) > 16384:
            raise ValueError("prompt exceeds 16384 UTF-8 bytes")
        if lock.locked():
            raise RuntimeError("inference service is busy; retry after the active call")
        async with lock:
            command = [str(binary), "generate", "-m", str(model_dir),
                       "--json", "--prompt-stdin", "-n", str(max_tokens),
                       "-t", str(temperature), "-j", str(threads)]
            if quantize:
                command.append("-q")
            if raw:
                command.append("--raw")
            start = time.perf_counter()
            process = None

            async def run_engine():
                nonlocal process
                process = await asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                return await process.communicate(encoded)

            try:
                stdout, stderr = await asyncio.wait_for(run_engine(), timeout)
            except TimeoutError as exc:
                raise RuntimeError(f"inference exceeded {timeout:g} seconds") from exc
            finally:
                # Cancellation and timeout must release the engine and its memory.
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
            latency_ms = (time.perf_counter() - start) * 1000
            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"engine failed (exit {process.returncode}): {detail}")
            result = GenerationResult(
                **json.loads(stdout), latency_ms=latency_ms,
                prompt_sha256=hashlib.sha256(encoded).hexdigest(),
                engine_sha256=engine_sha256, model_dir=str(model_dir),
                max_tokens=max_tokens, temperature=temperature, seed=42, raw=raw,
                threads=threads, quantized=quantize)
            if result.output_tokens > max_tokens or result.prompt_tokens > 2048:
                raise RuntimeError("engine returned token counts above request limits")
            return result

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/mk-infer"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        server = create_server(args.binary, args.model, threads=args.threads,
                               quantize=args.quantize, timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
