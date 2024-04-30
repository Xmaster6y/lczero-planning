.PHONY: checks
checks:
	poetry run pre-commit run --all-files

.PHONY: assets
assets:
	bash assets/resolve-assets.sh

.PHONY: demo
demo:
	cd lczero-planning-demo && poetry run python app.py
