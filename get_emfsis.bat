@echo off
setlocal EnableExtensions

set "WGET=C:\Windows\System32\wget.exe"
set "BASE=https://space.physics.uiowa.edu/emfisis/data/index"

cd /d "%~dp0"

echo Using wget: "%WGET%"
echo Destination: "%CD%"
echo.

for %%U in (
  "%BASE%/RBSP-A/L2/"
  "%BASE%/RBSP-B/L2/"
) do (
  echo ==================================================
  echo Downloading from: %%~U
  echo ==================================================

  "%WGET%" -r -np -nH --cut-dirs=3 -c ^
    -A "*.cdf" ^
    --tries=20 --timeout=30 --waitretry=5 ^
    "%%~U"
)

echo.
echo Cleaning up directory listing files...
del /s /q "*.html" "*.htm" "*.tmp" 2>nul

echo.
echo Done.
pause
endlocal