@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM get_magephem_one_day.bat
REM Download RBSP MagEphem definitive files for a given day
REM by parsing the year directory index and downloading matches.
REM
REM Usage:
REM   get_magephem_one_day.bat 2012-11-05
REM   get_magephem_one_day.bat RBSP-A 2012-11-05
REM   get_magephem_one_day.bat RBSP-B 2012-11-05
REM ============================================================

set "WGET=C:\Windows\System32\wget.exe"

pushd "%~dp0"

REM --- args: either [date] or [spacecraft date] ---
set "SC=%~1"
set "DATE=%~2"
if "%DATE%"=="" (
  set "DATE=%~1"
  set "SC=RBSP-A"
)

if "%DATE%"=="" (
  echo Usage: %~nx0 ^<RBSP-A^|RBSP-B^> YYYY-MM-DD
  echo    or: %~nx0 YYYY-MM-DD
  popd
  exit /b 1
)

REM --- map spacecraft to server token ---
set "BIRD="
if /I "%SC%"=="RBSP-A" set "BIRD=rbspa"
if /I "%SC%"=="RBSP-B" set "BIRD=rbspb"
if "%BIRD%"=="" (
  echo Spacecraft must be RBSP-A or RBSP-B. Got: "%SC%"
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
set "BASE=https://rbsp-ect.newmexicoconsortium.org/data_pub/%BIRD%/MagEphem/definitive"
set "YEARURL=%BASE%/%YYYY%/"
set "DEST=%CD%\%SC%\MagEphem\%YYYY%\%MM%\%DD%"

mkdir "%DEST%" 2>nul

echo.
echo [INFO] Spacecraft : %SC%
echo [INFO] Date       : %YYYY%-%MM%-%DD%
echo [INFO] Year URL   : %YEARURL%
echo [INFO] Dest       : "%DEST%"
echo.

REM --- download matching files directly via wget spider ---
echo [INFO] Downloading MagEphem files for %YMD%...
"%WGET%" -q --no-check-certificate -r -l1 -nd -np ^
  -A "*%YMD%*.h5,*%YMD%*.txt" ^
  -P "%DEST%" ^
  "%YEARURL%"

REM Check if any files were downloaded
dir /b "%DEST%\*%YMD%*.h5" >nul 2>&1
if errorlevel 1 (
  echo [WARN] No MagEphem .h5 files downloaded for %YMD%.
  popd
  exit /b 3
)

echo [OK] Done. Downloaded files in %DEST%:
dir /b "%DEST%\*%YMD%*.h5"

popd
endlocal
exit /b 0
