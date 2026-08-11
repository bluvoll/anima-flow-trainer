@echo off
setlocal EnableDelayedExpansion
rem Anima trainer installer -- Windows.
rem
rem   install.bat              install into .\venv and write .\start-gui.bat
rem   install.bat --recreate   delete an existing .\venv first
rem
rem Creates `venv\` (not `.venv\`) deliberately: the repo's own development environment is `.venv`,
rem managed by uv against uv.lock, and an installed copy must not silently take it over.

cd /d "%~dp0"

set "VENV=venv"
rem The diffusers commit this trainer was verified against. Anima support only exists on main, so
rem there is no release to pin to -- but floating HEAD means an install can break without a single
rem local change. Matches uv.lock; bump both together after re-running the parity gates.
set "DIFFUSERS_REF=50e7158093710f9c1b4ea9ff100137a91c9228f3"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "RECREATE=0"

if /i "%~1"=="--recreate" set "RECREATE=1"
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--help" goto :help

rem ---------------------------------------------------------------- find an interpreter
rem 3.11 first: every measurement in the README was taken on it. 3.12 resolves cleanly (torch
rem ships cp312 win_amd64 wheels) but has not been run, so it is a fallback rather than a peer.
rem 3.13 is excluded -- sdnq and triton are untested there.
rem Written as subroutine calls rather than nested for/if/&& blocks on purpose: cmd parses a whole
rem parenthesised block before running any of it, so a `&&` inside two levels of parens behaves in
rem ways that depend on the contents. `call` sidesteps the parser entirely.
set "PYTHON="
call :try_py 3.11
call :try_py 3.12
call :try_exe python3.11.exe
call :try_exe python3.12.exe
call :try_exe python.exe
if not defined PYTHON (
    echo.
    echo xxx No Python 3.11 or 3.12 found.
    echo     Install one from https://www.python.org/downloads/ and re-run.
    echo     During setup, tick "Add python.exe to PATH".
    echo     3.11 is the tested version; 3.12 also works.
    exit /b 1
)

rem `-V` rather than a `-c` one-liner: for /f delimits its command with single quotes, so a Python
rem snippet containing its own quotes is parsed by cmd in ways that depend on the contents. `-V`
rem prints "Python 3.11.15" with no quoting to get wrong.
for /f "tokens=2" %%V in ('%PYTHON% -V 2^>^&1') do set "PY_VER=%%V"
echo ==^> Python !PY_VER!  (%PYTHON%)
echo !PY_VER! | findstr /b "3.12." >nul && (
    echo !!! 3.12 resolves but has never been run end to end; 3.11 is the tested version.
)

rem ---------------------------------------------------------------- venv
if "%RECREATE%"=="1" if exist "%VENV%" (
    echo ==^> removing existing %VENV%\
    rmdir /s /q "%VENV%"
)
if exist "%VENV%\Scripts\python.exe" (
    echo ==^> reusing existing %VENV%\  ^(--recreate to start clean^)
) else (
    echo ==^> creating %VENV%\
    %PYTHON% -m venv "%VENV%" || (
        echo xxx could not create the venv
        exit /b 1
    )
)
set "VPY=%CD%\%VENV%\Scripts\python.exe"
if not exist "%VPY%" (
    echo xxx venv looks broken: no %VPY%
    exit /b 1
)

rem ---------------------------------------------------------------- dependencies
where uv >nul 2>&1
if not errorlevel 1 (
    rem uv honours uv.lock, so this reproduces the exact resolved set including the git diffusers
    rem and the cu128 torch index. UV_PROJECT_ENVIRONMENT points it at venv\ rather than .venv\.
    echo ==^> installing with uv ^(from uv.lock^)
    set "UV_PROJECT_ENVIRONMENT=%VENV%"
    uv sync --extra gui --extra ot || (
        echo xxx uv sync failed
        exit /b 1
    )
) else (
    echo ==^> installing with pip ^(uv not found^)
    "%VPY%" -m pip install --upgrade pip setuptools wheel >nul

    rem `requirements.txt` is `uv export` of uv.lock, so this path is as reproducible as the uv
    rem one: exact torch==2.10.0+cu128, the pinned diffusers commit, and the gui/ot extras.
    rem Resolving fresh from pyproject instead drifted -- measured transformers 5.15.0 against the
    rem locked 5.14.1. --extra-index-url, not --index-url: the +cu128 wheels live on PyTorch's
    rem index while everything else comes from PyPI.
    if exist requirements.txt (
        echo     from requirements.txt ^(pinned, matches uv.lock^)
        "%VPY%" -m pip install -r requirements.txt --extra-index-url "%TORCH_INDEX%" || (
            echo xxx pip install failed -- is git installed and on PATH?
            exit /b 1
        )
    ) else (
        echo !!! requirements.txt missing -- resolving fresh, which may not match uv.lock
        "%VPY%" -m pip install "torch==2.10.0" torchvision --index-url "%TORCH_INDEX%" || (
            echo xxx torch install failed
            exit /b 1
        )
        "%VPY%" -m pip install "diffusers @ git+https://github.com/huggingface/diffusers@%DIFFUSERS_REF%" || (
            echo xxx diffusers install failed -- is git installed and on PATH?
            exit /b 1
        )
        "%VPY%" -m pip install -e ".[gui,ot]" || (
            echo xxx install failed
            exit /b 1
        )
    )
)

rem ---------------------------------------------------------------- launchers
rem `convert_model` opens its window when given no arguments, so both launchers forward %* rather
rem than hardcoding --gui: a bare double-click still opens the GUI, and CLI flags keep working.
call :launcher start-gui.bat       anima.gui                 "Start the Anima trainer GUI."
call :launcher start-converter.bat anima.tools.convert_model "Convert single-file Anima checkpoints into a diffusers repo."

rem ---------------------------------------------------------------- report
echo.
echo ==^> checking the install
"%VPY%" "%~dp0anima\tools\check_install.py" --require-gui
if errorlevel 1 exit /b 1

echo.
echo ==^> done.  Start the GUI with:  start-gui.bat
echo     convert a single-file checkpoint:  start-converter.bat
echo     or the CLI:  venv\Scripts\python.exe -m anima.training.train configs\your.toml
exit /b 0

:help
echo install.bat              install into .\venv, write start-gui.bat and start-converter.bat
echo install.bat --recreate   delete an existing .\venv first
exit /b 0

rem ---------------------------------------------------------------- subroutines
rem Write one launcher.  %~1 file, %~2 module, %~3 description (quotes stripped by %~).
rem `pause` on failure so a double-clicked launcher does not vanish before the error is readable.
:launcher
echo ==^> writing %~1
(
    echo @echo off
    echo rem %~3  Generated by install.bat -- safe to delete and regenerate.
    echo cd /d "%%~dp0"
    echo "%%~dp0venv\Scripts\python.exe" -m %~2 %%*
    echo if errorlevel 1 pause
) > "%~1"
exit /b 0

rem The `py` launcher is the reliable way to reach a specific version on Windows: it is installed
rem with Python itself and knows about every version present, whether or not any of them is on PATH.
:try_py
if defined PYTHON exit /b 0
py -%~1 -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON=py -%~1"
exit /b 0

rem Fallback for machines without the launcher. Checks the version rather than trusting the name --
rem `python.exe` is whatever happens to be first on PATH, which is frequently 3.13 these days.
:try_exe
if defined PYTHON exit /b 0
where %~1 >nul 2>&1
if errorlevel 1 exit /b 0
%~1 -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
set "PYTHON=%~1"
exit /b 0
