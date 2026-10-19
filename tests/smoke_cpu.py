"""Check a built or installed CPU engine against deterministic toy-model output."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from model_fixture import write_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    expected = {"text": "AAA", "prompt_tokens": 3, "output_tokens": 3,
                "finish_reason": "length"}
    with tempfile.TemporaryDirectory(prefix="vnni-smoke-") as directory:
        model = Path(directory)
        write_model(model)
        for quantized in (False, True):
            for threads in (1, 2):
                command = [str(binary), "generate", "-m", str(model), "--raw",
                           "--json", "--prompt-stdin", "-n", "3", "-t", "0",
                           "-j", str(threads)]
                if quantized:
                    command.append("-q")
                for repeat in range(2):
                    result = subprocess.run(command, input="hé", text=True, encoding="utf-8",
                                            capture_output=True, timeout=2, check=True)
                    output = json.loads(result.stdout)
                    actual = {key: output[key] for key in expected}
                    if actual != expected:
                        raise RuntimeError(f"{command}: expected {expected}, got {actual}")
                    print(f"{'Q4' if quantized else 'BF16'} threads={threads} "
                          f"run={repeat + 1}: fixed output passed", flush=True)
    print("CPU smoke passed: text=AAA, prompt_tokens=3, output_tokens=3, finish_reason=length")


if __name__ == "__main__":
    main()
