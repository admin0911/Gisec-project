## Active web scanner

The MNIST and CIFAR-10 scan adapters now use the teammate pipeline through
`web_noise.scan_noise` and `scan_as_connector_result`, after patch scanning.
All supplied classes are scanned; attack target and poison identities are not passed.
The residual-signature method falls back to background-lift when no residual finding is accepted.
Shared flags enter optional human review and training preparation. Saved Keep overrides flags;
unreviewed flags are quarantined; saved Unsure stays unresolved and excluded.
Cached scans are invalidated by changes to the noise implementation.

The retired consensus-pixel implementation is available in Git history.

Quick integration check: 1,000 randomly selected training images per dataset, seed 2026,
5% shared blended noise, alpha 0.1, attack seed 0, target 0. MNIST caught 50/50 with
0 false positives; CIFAR caught 0/50 with 0 false positives. Both clean checks had
0 flags. These are development checks, not held-out validation or ASR measurements.
