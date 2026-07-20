.PHONY: check campaign-test campaign-generate

campaign-test:
	PYTHONPATH=campaigns/one-layer-deeper/src python -m unittest discover -s campaigns/one-layer-deeper/tests -v

campaign-generate:
	PYTHONPATH=campaigns/one-layer-deeper/src python -m old_campaign.cli generate \
		--profile campaigns/one-layer-deeper/profiles/baseline_adamw.json \
		--template campaigns/one-layer-deeper/templates/submission.py.tmpl \
		--output /tmp/iterant-baseline-submission.py

check: campaign-test campaign-generate
