.PHONY: help lint ci ci-security pr-attribution-check

.DEFAULT_GOAL := help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-24s %s\n", $$1, $$2}'

lint: ## Run repository lint checks
	@pre-commit run --all-files

ci-security: ## Run local security parity checks (Trivy)
	@command -v trivy >/dev/null 2>&1 || (echo "Install trivy: https://trivy.dev/latest/getting-started/installation/" && exit 1)
	@trivy fs --severity HIGH,CRITICAL --exit-code 1 .

pr-attribution-check: ## Check branch commits and optional PR text for forbidden attribution
	@chmod +x .github/scripts/attribution-guard.sh
	@.github/scripts/attribution-guard.sh \
		--base-ref origin/main \
		$${PR_TITLE:+--title "$${PR_TITLE}"} \
		$${PR_BODY_FILE:+--body-file "$${PR_BODY_FILE}"}

ci: lint ci-security pr-attribution-check ## Run local CI parity checks before PR
	@echo "Local CI checks passed."
