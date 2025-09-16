<#
Create a git bundle of the repository for offline transfer.
Usage:
  powershell -ExecutionPolicy Bypass -File .\make_bundle.ps1 -Out repo.bundle
#>
param(
    [string]$Out = "repo.bundle"
)

Write-Host "Creating git bundle -> $Out"

# Ensure we're in repository root
if (-not (Test-Path .git)) {
    Write-Error "This script must be run from the repository root (where .git exists)."
    exit 1
}

git bundle create $Out --all
if ($LASTEXITCODE -ne 0) {
    Write-Error "git bundle failed"
    exit $LASTEXITCODE
}

Write-Host "Bundle created: $Out"
