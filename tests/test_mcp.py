"""Real engine/MCP checks with generated toy weights; no model downloads."""

import asyncio
import hashlib
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vnni_mcp import create_server

logging.getLogger().setLevel(logging.WARNING)


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/mk-infer"


def write_model(directory):
    """One-layer, 32-wide fixture whose greedy output is always ASCII A."""
    config = {"model_type": "llama", "hidden_size": 32, "intermediate_size": 32,
              "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
              "vocab_size": 259, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
              "tie_word_embeddings": True, "eos_token_id": 256}
    (directory / "config.json").write_text(json.dumps(config))
    extra, vocab = 0, {}
    for byte in range(256):
        printable = 33 <= byte <= 126 or 161 <= byte <= 172 or byte >= 174
        vocab[chr(byte if printable else 256 + extra)] = byte
        if not printable:
            extra += 1
    specials = [{"id": 256, "content": "<|endoftext|>"},
                {"id": 257, "content": "<|im_start|>"},
                {"id": 258, "content": "<|im_end|>"}]
    (directory / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": vocab, "merges": []}, "added_tokens": specials}))
    tensors, payload = {}, bytearray()

    def add(name, shape, values):
        start = len(payload)
        for value in values:
            payload.extend(struct.pack("<f", value)[2:])  # BF16
        tensors[name] = {"dtype": "BF16", "shape": shape,
                         "data_offsets": [start, len(payload)]}

    add("model.embed_tokens.weight", [259, 32],
        [(2.0 if row == 65 else 1.0) if col == 0 else 0.0
         for row in range(259) for col in range(32)])
    add("model.norm.weight", [32], [1.0] * 32)
    for name in ("input_layernorm", "post_attention_layernorm"):
        add(f"model.layers.0.{name}.weight", [32], [1.0] * 32)
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        add(f"model.layers.0.{name}.weight", [32, 32], [0.0] * 1024)
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        write_model(self.model)

    def engine(self, prompt, *args):
        return subprocess.run([str(BINARY), "generate", "-m", str(self.model),
                               "--prompt-stdin", *args], input=prompt.encode(),
                              capture_output=True, timeout=3, check=False)

    async def test_native_json_exact_tokens_and_legacy_output(self):
        for quantize in ([], ["-q"]):
            reply = self.engine("hé", "--raw", "--json", "-n", "3", *quantize)
            self.assertEqual(reply.returncode, 0, reply.stderr)
            result = json.loads(reply.stdout)
            self.assertEqual((result["text"], result["prompt_tokens"], result["output_tokens"]),
                             ("AAA", 3, 3))
            self.assertEqual(result["finish_reason"], "length")
            self.assertGreater(result["generation_ms"], 0)
        legacy = self.engine("a", "--raw", "-n", "2")
        self.assertEqual((legacy.returncode, legacy.stdout), (0, b"AA"))
        config_file = self.model / "config.json"
        config = json.loads(config_file.read_text())
        config["eos_token_id"] = 65
        config_file.write_text(json.dumps(config))
        stopped = json.loads(self.engine("a", "--raw", "--json", "-n", "3").stdout)
        self.assertEqual((stopped["output_tokens"], stopped["text"], stopped["finish_reason"]),
                         (0, "", "stop"))

    async def test_native_rejects_bounds_before_generation(self):
        cases = [("", ["--raw"]), ("a", ["-n", "0"]), ("a", ["-n", "1025"]),
                 ("a" * 2049, []), ("a" * 16385, []), ("a", ["-t", "nan"])]
        for prompt, args in cases:
            with self.subTest(args=args, prompt_length=len(prompt)):
                reply = self.engine(prompt, "--json", *args)
                self.assertEqual(reply.returncode, 1)
                self.assertEqual(reply.stdout, b"")
        config_file = self.model / "config.json"
        config = json.loads(config_file.read_text())
        config["max_position_embeddings"] = 8
        config_file.write_text(json.dumps(config))
        self.assertEqual(self.engine("abcdefg", "--json", "--raw", "-n", "2").returncode, 1)

    async def test_mcp_stdio_schema_results_and_errors(self):
        params = StdioServerParameters(command=sys.executable, args=[
            "-m", "vnni_mcp", "--binary", str(BINARY), "--model", str(self.model)])
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=5)) as session,
        ):
            await session.initialize()
            tools = (await session.list_tools()).tools
            self.assertEqual([tool.name for tool in tools], ["generate"])
            self.assertEqual(tools[0].inputSchema["properties"]["max_tokens"]["maximum"], 1024)
            self.assertIn("latency_ms", tools[0].outputSchema["properties"])
            request = {"prompt": "hé", "max_tokens": 2, "raw": True}
            first = await session.call_tool("generate", request)
            self.assertFalse(first.isError)
            result = first.structuredContent
            self.assertEqual((result["text"], result["prompt_tokens"], result["output_tokens"]),
                             ("AA", 3, 2))
            self.assertGreaterEqual(result["latency_ms"], result["generation_ms"])
            self.assertEqual(result["prompt_sha256"], hashlib.sha256("hé".encode()).hexdigest())
            self.assertEqual(result["engine_sha256"], hashlib.sha256(BINARY.read_bytes()).hexdigest())
            self.assertEqual((result["seed"], result["threads"], result["quantized"]), (42, 1, False))
            second = await session.call_tool("generate", request)
            self.assertEqual(second.structuredContent["text"], result["text"])
            templated = await session.call_tool("generate", {"prompt": "hé", "max_tokens": 1})
            self.assertGreater(templated.structuredContent["prompt_tokens"], 3)
            for invalid in ({"max_tokens": 0}, {"max_tokens": 1025}, {"max_tokens": True},
                            {"prompt": ""}, {"prompt": "😀" * 5000}, {"temperature": float("inf")}):
                reply = await session.call_tool("generate", request | invalid)
                self.assertTrue(reply.isError, invalid)

    async def test_timeout_cancellation_busy_and_process_cleanup(self):
        # A controlled stalled executable tests process lifecycle, not inference metrics.
        stalled = self.model / "stalled-engine"
        pid_file = self.model / "pid"
        stalled.write_text(f"#!{sys.executable}\nimport os, time\n"
                           f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
                           "time.sleep(10)\n")
        stalled.chmod(0o700)
        server = create_server(stalled, self.model, timeout=0.2)
        with self.assertRaisesRegex(Exception, "exceeded"):
            await server.call_tool("generate", {"prompt": "a"})
        pid = int(pid_file.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        server = create_server(stalled, self.model, timeout=2)
        task = asyncio.create_task(server.call_tool("generate", {"prompt": "a"}))
        await asyncio.sleep(0.1)
        with self.assertRaisesRegex(Exception, "busy"):
            await server.call_tool("generate", {"prompt": "b"})
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)

    async def test_engine_failure_and_bad_output(self):
        for body, message in [("import sys; sys.exit(7)", "exit 7"),
                              ("print('invalid json')", "Error")]:
            script = self.model / "broken-engine"
            script.write_text(f"#!{sys.executable}\n{body}\n")
            script.chmod(0o700)
            with self.assertRaisesRegex(Exception, message):
                await create_server(script, self.model).call_tool("generate", {"prompt": "a"})
        with self.assertRaisesRegex(ValueError, "build mk-infer"):
            create_server(self.model / "missing", self.model)
        with self.assertRaisesRegex(ValueError, "timeout"):
            create_server(BINARY, self.model, timeout=float("nan"))


if __name__ == "__main__":
    unittest.main()
