$ErrorActionPreference = 'Stop'

if (-not [string]::IsNullOrWhiteSpace((git status --porcelain))) {
    throw "Repository is not clean. Please commit or stash changes."
}

git switch master
if ($LASTEXITCODE -ne 0) { throw "Failed to switch to master branch." }

git pull
if ($LASTEXITCODE -ne 0) { throw "Failed to pull latest changes from remote." }

if (-not (Test-Path -Path ".venv" -PathType Container)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create virtual environment." }
}

$activate = Join-Path -Path ".venv" -ChildPath "Scripts\Activate.ps1"
if (Test-Path $activate) {
    . $activate
} else {
    pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "Failed to install development dependencies." }
}

pip install --upgrade --upgrade-strategy eager -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install development dependencies." }

pytest
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

$tag = "tested-" + (Get-Date -Format "yyyyMMddHHmmss")
git tag $tag
if ($LASTEXITCODE -ne 0) { throw "Failed to create git tag." }

git push origin $tag
if ($LASTEXITCODE -ne 0) { throw "Failed to push git tag to remote." }
