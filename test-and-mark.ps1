$ErrorActionPreference = 'Stop'

# if (-not [string]::IsNullOrWhiteSpace((git status --porcelain))) {
#     Write-Error "Repository is not clean. Please commit or stash changes."
#     exit 1
# }

git switch master
git pull

if (-not (Test-Path -Path ".venv" -PathType Container)) {
    python -m venv .venv
}

$activate = Join-Path -Path ".venv" -ChildPath "Scripts\Activate.ps1"
if (Test-Path $activate) {
    . $activate
} else {
    pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install failed with exit code $LASTEXITCODE. Aborting tests."
        exit $LASTEXITCODE
    }
}

pip install -r requirements-dev.txt

pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$tag = "tested-" + (Get-Date -Format "yyyyMMddHHmmss")
git tag $tag
git push origin $tag
