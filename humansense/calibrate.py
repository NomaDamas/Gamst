"""Fit and validate an empirical score calibration from two labeled JSONL files."""
import argparse
import json
from pathlib import Path
from .feedback import LabeledOutcome, fit_calibration


def read_labels(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) != {'id', 'score', 'outcome'}:
                    raise ValueError('Expected exactly id, score, outcome')
                yield LabeledOutcome(**item)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--training-dataset', required=True)
    parser.add_argument('--validation-dataset', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--scorer-model-id', required=True)
    parser.add_argument('--scorer-model-version', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.training).resolve() == Path(args.validation).resolve():
        parser.error('Training and validation must be different files')
    if Path(args.output).exists():
        parser.error('Output exists; choose a new artifact path')
    if args.training_dataset == args.validation_dataset:
        parser.error('Training and validation datasets must differ')
    try:
        artifact = fit_calibration(target=args.target, training=read_labels(args.training),
            validation=read_labels(args.validation), training_dataset=args.training_dataset,
            validation_dataset=args.validation_dataset, scorer_model_id=args.scorer_model_id,
            scorer_model_version=args.scorer_model_version)
        # Exclusive creation protects labels or a prior model from accidental overwrite.
        with Path(args.output).open('x', encoding='utf-8') as handle:
            handle.write(artifact.to_json()+'\n')
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps({'artifact': str(args.output), 'evaluation': artifact.evaluation}, indent=2))
    return 0 if artifact.validated else 1


if __name__ == '__main__':
    raise SystemExit(main())
