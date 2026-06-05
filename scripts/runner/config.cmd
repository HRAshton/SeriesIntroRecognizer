@echo off
REM Usage: run-runner-in-podman.cmd --url https://github.com/HRAshton/series-intro-recognizer --token ABCDEFG
REM See https://github.com/HRAshton/series-intro-recognizer/settings/actions/runners/new

setlocal enabledelayedexpansion

set "imageName=hra-cuda-runner:latest"
set "runnerName=github-runner"
set "url="
set "token="

REM Parse command line arguments
:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--url" (
    set "url=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--token" (
    set "token=%~2"
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:args_done
echo Building image %imageName%
podman build -t %imageName% .
if errorlevel 1 (
    echo ERROR: Failed to build image
    exit /b 1
)
echo Built image %imageName%

echo Removing any existing container named %runnerName%
podman stop %runnerName% >nul 2>&1
podman rm %runnerName% >nul 2>&1
if errorlevel 1 (
    echo No existing container named %runnerName% to remove
) else (
    echo Removed existing container %runnerName%
)

echo Running container %runnerName% with image %imageName%. URL: %url%; Token: %token%
podman run ^
    -d ^
    --name %runnerName% ^
    --device nvidia.com/gpu=all ^
    -e RUNNER_NAME=%runnerName% ^
    -e GITHUB_URL=%url% ^
    -e GITHUB_TOKEN=%token% ^
    -v %CD%/entrypoint.sh:/entrypoint.sh ^
    %imageName% ^
    /entrypoint.sh

endlocal

