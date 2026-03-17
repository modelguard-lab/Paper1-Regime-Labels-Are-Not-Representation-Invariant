@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo Building manuscript...

pdflatex -interaction=nonstopmode main.tex >nul 2>&1
bibtex main
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
pdflatex -interaction=nonstopmode main.tex >nul 2>&1

if not exist main.pdf (
    echo ERROR: PDF was not generated. Check main.log for errors.
    exit /b 1
)

for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value 2^>nul') do set "dt=%%a"
if not defined dt for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "dt=%%a"
set "datestr=!dt:~0,8!"
set "outname=Paper1_Manuscript_FRL_!datestr!.pdf"

copy /y main.pdf "!outname!" >nul
echo Generated: !outname!

del main.aux main.log main.out main.bbl main.blg main.spl main.synctex.gz main.fls main.fdb_latexmk 2>nul
del main.pdf 2>nul

echo Done. Output: !outname!
endlocal
pause
