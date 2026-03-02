# ml-production-pipeline

Repository bootstrap for model training, serving, and drift monitoring workflows.

## Local CI Gate

Run before every PR:

```bash
make ci
```

This runs:
- formatting and lint hooks via pre-commit
- local security scan parity (`trivy fs`)
- attribution guard checks on branch commits (and optional PR text)
