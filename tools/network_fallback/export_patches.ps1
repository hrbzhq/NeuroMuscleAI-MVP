<#
Export local commits not on origin/master as patch files.
Usage:
  powershell -ExecutionPolicy Bypass -File .\export_patches.ps1 -OutDir patches
#>
param(
    [string]$OutDir = "patches"
)

if (-not (Test-Path .git)) {
    Write-Error "This script must be run from the repository root (where .git exists)."
    exit 1
}

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# Determine commits ahead of origin/master
$range = "origin/master..HEAD"
Write-Host "Exporting patches for range: $range to $OutDir"

git fetch origin master

git format-patch $range -o $OutDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "git format-patch failed"
    exit $LASTEXITCODE
}

Write-Host "Patches exported to: $OutDir"
