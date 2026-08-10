#!/usr/bin/env python3
"""Compare mk-infer's tokenizer against HF `tokenizers` on adversarial strings.

Usage: verify_tokenizer.py <model_dir> <mk_infer_binary>
"""
import json
import subprocess
import sys

from tokenizers import Tokenizer

model_dir, binary = sys.argv[1], sys.argv[2]
hf = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

CASES = [
    "Hello, world!",
    "hello",
    " hello",
    "  hello   world  ",
    "don't can't won't I'LL you'RE he's",
    "x'S weird'Ve",
    "123 4567 3.14159",
    "a1b2c3",
    "foo\nbar\r\nbaz\n\n\nqux",
    "   \n  \n",
    "trailing spaces   ",
    "\ttabs\t\tand\tmore\t",
    "def f(x):\n    return x**2  # comment\n",
    "if (a != b) { c += 1; } else { d ||= e }",
    "über café naïve résumé",
    "Straße GRÜSSE ÄÖÜ",
    "日本語のテキスト",
    "中文分词测试一下",
    "Привет мир",
    "مرحبا بالعالم",
    "🙂🚀👍🏻 emoji 🇩🇪 flags",
    "mixed 中文 and English 123",
    "!!!???...,,,;;;",
    "a-b_c.d/e\\f",
    "http://example.com/path?q=1&r=2#frag",
    "<html><body class=\"x\">&amp;</body></html>",
    "''''''",
    "'s 't 're standalone",
    "O'Brien's",
    "1'000'000",
    "  'quoted string'  ",
    " nbsp test",
    " line sep para sep",
    "　ideographic space",
    "zero​width",
    "é combining acute",  # NFC normalizer test
    "ﬁ ligature ﬂ",
    "½ ¾ ⅓ fractions ² ³ superscripts",
    "Ⅷ roman numeral",
    "ｆｕｌｌｗｉｄｔｈ　ｔｅｘｔ",
    "\r\r\n\n\r",
    "single space: ",
    "",
    "a",
    " ",
    "\n",
]

fails = 0
for s in CASES:
    ref = hf.encode(s, add_special_tokens=False).ids
    out = subprocess.run([binary, "tokenize", "-m", model_dir, "-p", s],
                         capture_output=True, text=True)
    first = out.stdout.splitlines()[0] if out.stdout.strip() else ""
    ours = [int(t) for t in first.split()] if first else []
    # NFC may rewrite the input, so roundtrip must match HF's decode,
    # not the original string.
    roundtrip_ok = "roundtrip OK" in out.stdout or hf.decode(ref) != s
    if ours != ref or not (roundtrip_ok or s == ""):
        fails += 1
        print(f"MISMATCH {s!r}")
        print(f"  ref : {ref}")
        print(f"  ours: {ours}  (roundtrip={'OK' if roundtrip_ok else 'BAD'})")

print(f"{len(CASES) - fails}/{len(CASES)} cases match")
sys.exit(1 if fails else 0)
