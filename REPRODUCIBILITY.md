# Reproducibility checklist

Before reporting or releasing a result:

1. Record the Git commit, Python/PyTorch/Transformers versions, model revision,
   device, dtype, prompt count, prompt seeds, output position, vocabulary mode,
   and ablation mode.
2. Keep detection, calibration, and evaluation splits disjoint where the script
   defines separate split seeds.
3. Use `--top-r 0` for the GPT-2-small headline protocol. Do not compare a top-r
   approximation to the full-vocabulary paper number without labeling it.
4. Preserve the JSON artifact emitted by the script; do not copy rounded README
   numbers into a result file.
5. Run `python scripts/check_release.py` before publishing.

The dependency versions used for the release checks are recorded in
`requirements-tested.txt`.
