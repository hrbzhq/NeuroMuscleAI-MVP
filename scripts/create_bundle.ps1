Param()

# Create a git bundle of the repository (all refs)
Write-Output "Creating git bundle repo.bundle..."
git bundle create repo.bundle --all
if ($LASTEXITCODE -ne 0) {
    Write-Error "git bundle creation failed"
    exit 1
}
Write-Output "Bundle created: repo.bundle"
Write-Output "SHA256:"
Get-FileHash repo.bundle -Algorithm SHA256
