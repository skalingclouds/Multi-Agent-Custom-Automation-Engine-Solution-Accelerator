# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2026-01-04 13:28]
Azure CLI (az) and Bicep CLI are not available in the sandboxed environment - use manual syntax verification as fallback for Bicep template validation

_Context: subtask-10-3: Verify Bicep template validates correctly. The sandbox environment does not have Azure CLI or Bicep CLI installed. Manual pattern-based verification against existing templates in main.bicep was used as an alternative._
