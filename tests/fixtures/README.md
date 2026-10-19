# Fixed engine references

`tokenizer.json` contains a synthetic byte-level vocabulary. IDs 0–255 equal
their source byte values under the GPT-2 byte mapping. IDs 256–273 follow the
listed merge order. IDs 300 and 301 represent overlapping special strings.
The merge list uses both string and pair-array encodings.

`regression.json` specifies exact token IDs and decoded UTF-8 strings. Cases
cover merge priority, leftmost ties, leading spaces, NFC normalization, emoji,
CJK, contractions, individual digits, CRLF, punctuation, and special-token
handling. Deliberate merge candidates cross pre-token boundaries so a split
regression changes the expected IDs. Decoding the decomposed accent case must
produce NFC text.

The quantization fixtures specify 32-element inputs, Q4 scale bits, packed
bytes, decoded values, Q8 scale bits, signed values, and sums. These references
come from the format equations and explicit integer levels. They were not
captured from an engine run or a production model. The cases cover zero,
positive/negative tied maxima, clipping, half-away-from-zero rounding, both
nibble halves, and multiple block scales.

Q4 stores indices 0–15 in low nibbles and indices 16–31 in high nibbles.
Decoding uses `d * (nibble - 8)`. The first maximum by absolute value sets
`d = maximum / -8`. The zero fixture preserves the negative-zero Q4 scale
bits. Q8 uses `d = abs_max / 127`, signed rounded levels, and their sum.
The engine truncates the stored half-precision scale. Q8 scale `0x2c08`
equals `129/2048`; `0x2808` equals half that value.

The nonzero Q4 integer levels and Q8 signed levels have integer dot product
11178. The five 64-column matvec rows select block pairs from these fixtures.
Their scale factors are respectively `0.5`, `-1`, `-1`, `-2`, and `1.5`,
multiplied by `11178 * 129/2048`. Both scalar reference and selected CPU kernels
must equal these fixed outputs, with and without bias, for row counts 1–5.
The binary fractions make these comparisons exact. The test checks each field
explicitly and does not compare struct padding or depend on `assert`/`NDEBUG`.

Run from any working directory through CTest:

```bash
ctest --test-dir build --output-on-failure --timeout 20
```

`tests/model_fixture.py` supplies the separate CPU smoke model and the MCP test
model. Its 32-wide, one-layer transformer has zero attention/MLP projections,
unit norms, and tied embeddings whose first coordinate favors ASCII `A`.
All other coordinates are zero. Every greedy decode step therefore emits `A`.
The tokenizer has no merges, so `hé` encodes to the three byte IDs 104, 195,
and 169. This fixture tests model loading and generation in BF16/Q4 modes.
It does not validate nonzero transformer activations against HuggingFace.

Change reference values only when an intended format or tokenizer contract
changes. Explain the new expected values from the input data and equations.
