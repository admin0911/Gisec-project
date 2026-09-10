# Active image pipeline connectors

- Build: `FeatureBundle` for CIFAR encoders; `ImageInputBundle` for original/post-attack pixels. MNIST currently skips encoders.
- Detector input: `detector_input` through `paired_feature_inputs` for CIFAR; `image_detector_input` for MNIST pixels. Both return `DetectorInput` (X, sample_ids, optional y), without poison truth.
- Detector output: `detector_result` validates the shared dictionary: detector_name, version, sample_ids, scores, flags, settings, evidence.
- Assessment: aligned sample IDs, assessment states and vote counts. MNIST suspects require two of three detector flags.
- Review and selection: saved review decisions keyed by sample ID feed `cleaning.selection.merge_choices`. Both MNIST preview and training preparation call this same connector. MNIST keeps unreviewed one-vote rows; CIFAR retains its own existing policy.
- Cleaning: frozen selection manifest (sample_ids, actions, reasons, policy and source hash). `partition_dataset` joins decisions by sample ID into dataset views.
- Training: `training_input` creates `TrainingInput` with dataset, IDs, version and split. The classifier receives current images/labels, not scores or poison identities. Clean reference and test data are benchmark-only inputs.
- Evaluation: known poison metadata is read separately, after selection. The dashboard reads API summaries; it does not implement detector or cleaning rules.

Archive metadata reads for hashing/alignment and JSON API serialization are transport operations, not alternate detector inputs. Experimental scripts may implement alternative policies; this map describes the active MNIST/CIFAR label-flip web pipeline, not unintegrated teammate detectors.
