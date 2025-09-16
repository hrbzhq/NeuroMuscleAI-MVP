<#
Purge history of specified paths using git-filter-repo.

USAGE (recommended): run on a mirrored clone because this rewrites history and will force-push.

1) Create a mirror clone (outside repo):
   git clone --mirror https://github.com/<owner>/<repo>.git repo-mirror.git
2) Run this script inside the mirror clone (it will modify the mirror):
   powershell -ExecutionPolicy Bypass -File purge_history.ps1 -PathsToRemove ".git_backup_20250915144245","data"
3) Review the result, then push back to origin (force):
   git push --force --all
   git push --force --tags

WARNING: This is destructive and will rewrite published history. All collaborators must re-clone after the rewrite.
#>
param(
    [Parameter(Mandatory=$true)]
    [string[]]$PathsToRemove
)

# Ensure git-filter-repo is installed
try {
    git filter-repo --version | Out-Null
} catch {
    Write-Host "git-filter-repo not available. Attempting to install via pip..."
    python -m pip install --upgrade git-filter-repo
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install git-filter-repo. Install it manually and rerun."
        exit 1
    }
}

# Construct arguments
$frArgs = @()
foreach ($p in $PathsToRemove) {
    $frArgs += "--path"
    $frArgs += $p
    $frArgs += "--invert-paths"
}

Write-Host "About to run: git filter-repo $($frArgs -join ' ')"
Write-Host "Make sure you are running in a mirror clone and you have a backup."

Read-Host -Prompt "Press Enter to continue (or Ctrl+C to abort)"

# Run filter-repo
git filter-repo @frArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "git filter-repo failed"
    exit $LASTEXITCODE
}

Write-Host "git-filter-repo finished. Review changes then run 'git push --force --all' and 'git push --force --tags' to update origin."
