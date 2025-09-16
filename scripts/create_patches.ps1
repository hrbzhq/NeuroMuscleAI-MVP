Param()

Write-Output "Generating format-patch series against origin/master..."
git fetch origin
if ($LASTEXITCODE -ne 0) { Write-Error "git fetch origin failed"; exit 1 }
$base = git rev-parse origin/master
if ($LASTEXITCODE -ne 0) { Write-Error "Could not resolve origin/master"; exit 1 }
git format-patch $base..HEAD -o patches
if ($LASTEXITCODE -ne 0) { Write-Error "format-patch failed"; exit 1 }
Write-Output "Patches created in ./patches"
