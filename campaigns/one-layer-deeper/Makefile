.PHONY: install test generate-baseline matrix gate

install:
	python -m pip install -e .

test:
	python -m unittest discover -s tests -v

generate-baseline:
	old-campaign generate --profile profiles/baseline_adamw.json --template templates/submission.py.tmpl --output artifacts/submissions/baseline_adamw/submission.py

matrix:
	old-campaign matrix --config configs/run_matrix.json

gate:
	old-campaign gate --policy governance/hard_gate_policy.json artifacts/evidence/*.json
