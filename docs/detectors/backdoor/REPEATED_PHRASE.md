# Repeated phrase detector (experimental)

`detectors/backdoor/repeated_phrase.py` accepts `TextInputBundle` from `poison_features/text_inputs.py`, containing only observed texts, integer labels and unique sample IDs. It returns the shared `detector_result` dictionary. No known trigger, attack target, poison rate or poison identities are detector inputs.

The heuristic mines 2–4 word phrases, with document support >= max(10, 0.2% of rows), <=20% of rows, dominant-label purity >=95%, label lift >=1.5 and >=80% sharing a token offset from the start or end. It checks all offsets and observed labels. These defaults are provisional, not calibrated probabilities. Natural repeated introductions, duplicates and templates can cause false positives; variable placement, paraphrases, single-word and syntactic triggers can be missed. Scores mean evidence strength and flags request review.

Run:

```powershell
python -m experiments.scan_imdb_phrases --prepared artifacts/imdb_backdoor_preparation/20260910-055921-075768
```

Initial 5,000-row development evaluation: clean-only 36 false flags (0.72%); 5% phrase attack 250/250 poisons found plus 20 false positives, 92.59% precision and 100% recall. This attack was previously inspected during development: these are not held-out generalization results. Truth is loaded after scoring, solely for evaluation. Synthetic tests cover alternative phrases and both targets but do not replace held-out real-data validation.

Outputs are individual clean/poisoned standard connector results and phrase-evaluation.json in the prepared directory. App builds now save post-attack text through TextInputBundle, and run phrase scanning after the three label-flip checks. Phrase flags join human review as unresolved findings. IMDB backdoor training via the app remains disabled; use the comparison script. Retraining after selection and ASR comparison remain a separate next step.
