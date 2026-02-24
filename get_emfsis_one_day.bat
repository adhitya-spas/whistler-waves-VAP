@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM get_emfsis_one_day.bat
REM Download RBSP EMFSIS L2 files for a given day
REM by parsing the day directory index and downloading matches.
REM
REM Usage:
REM   get_emfsis_one_day.bat 2012-11-05
REM   get_emfsis_one_day.bat RBSP-A 2012-11-05
REM   get_emfsis_one_day.bat RBSP-B 2012-11-05
REM ============================================================

set "WGET=C:\Windows\System32\wget.exe"
set "BASE=https://space.physics.uiowa.edu/emfisis/data/index"

pushd "%~dp0"

REM --- args: either [date] or [spacecraft date] ---
set "SC=%~1"
set "DATE=%~2"
if "%DATE%"=="" (
  set "DATE=%~1"
  set "SC="
)

if "%DATE%"=="" (
  echo Usage: %~nx0 ^<RBSP-A^|RBSP-B^> YYYY-MM-DD
  echo    or: %~nx0 YYYY-MM-DD
  popd
  exit /b 1
)

REM --- split YYYY-MM-DD -> YYYY MM DD ---
for /f "tokens=1-3 delims=-" %%a in ("%DATE%") do (
  set "YYYY=%%a"
  set "MM=%%b"
  set "DD=%%c"
)

if "%YYYY%"=="" (
  echo Bad date format. Use YYYY-MM-DD
  popd
  exit /b 1
)

set "YMD=%YYYY%%MM%%DD%"

REM --- Download for specified spacecraft or both ---
if "%SC%"=="" (
  set "SPACECRAFT=RBSP-A RBSP-B"
) else (
  set "SPACECRAFT=%SC%"
)

for %%S in (%SPACECRAFT%) do (
  set "DAYURL=%BASE%/%%S/L2/%YYYY%/%MM%/%DD%/"
  set "DEST=%CD%\%%S\L2\%YYYY%\%MM%\%DD%"
  
  mkdir "!DEST!" 2>nul

  echo.
  echo [INFO] Spacecraft : %%S
  echo [INFO] Date       : %YYYY%-%MM%-%DD%
  echo [INFO] Day URL    : !DAYURL!
  echo [INFO] Dest       : "!DEST!"
  echo.

  REM --- download matching CDF files directly via wget spider ---
  echo [INFO] Downloading EMFSIS L2 CDF files for %%S on %YMD%...
  "%WGET%" -q --no-check-certificate -r -l1 -nd -np ^
    -A "*.cdf" ^
    -P "!DEST!" ^
    "!DAYURL!"

  REM Check if any files were downloaded
  dir /b "!DEST!\*.cdf" >nul 2>&1
  if errorlevel 1 (
    echo [WARN] No EMFSIS L2 CDF files downloaded for %%S on %YMD%.
  ) else (
    echo [OK] Done. Downloaded files in !DEST!:
    dir /b "!DEST!\*.cdf"
  )

  echo Done: %%S
)

echo All done.
popd
endlocal
exit /b 0
