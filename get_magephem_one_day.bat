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

REM --- extract matching file URLs from the directory index ---
set "URLLIST=%TEMP%\magephem_urls_%BIRD%_%YMD%.txt"
if exist "%URLLIST%" del /f /q "%URLLIST%" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
  "$u='%YEARURL%'; $ymd='%YMD%'; $out='%URLLIST%';" ^
  "$r = Invoke-WebRequest -Uri $u -UseBasicParsing;" ^
  "$links = $r.Links | ForEach-Object { $_.href } | Where-Object { $_ -and ($_ -match $ymd) -and (($_ -match '\.h5$') -or ($_ -match '\.txt$')) } | Select-Object -Unique;" ^
  "if(-not $links){ exit 3 }" ^
  "$urls = $links | ForEach-Object { if($_ -match '^https?://'){ $_ } else { $u + $_ } };" ^
  "$urls | Set-Content -Encoding ascii -Path $out; exit 0"

if errorlevel 3 (
  echo [WARN] No MagEphem files found for %YMD% in the year index.
  popd
  exit /b 3
)

echo [INFO] URLs to download:
type "%URLLIST%"
echo.

REM --- download each URL directly ---
set "DLFAIL=0"
for /f "usebackq delims=" %%U in ("%URLLIST%") do (
  echo [INFO] wget %%U
  "%WGET%" -c -nd -P "%DEST%" "%%U"
  if errorlevel 1 set "DLFAIL=1"
)

echo.
if "%DLFAIL%"=="1" (
  echo [ERROR] One or more downloads failed.
  popd
  exit /b 5
)

echo [OK] Done. Downloaded files in:
dir /b "%DEST%\*%YMD%*.h5" 2>nul
dir /b "%DEST%\*%YMD%*.txt" 2>nul

popd
endlocal
exit /b 0
