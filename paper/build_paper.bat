@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================
echo Building Paper 1: Main Manuscript
echo ============================================================

echo.
echo Copying figures from ..\outputs ...
if exist "..\outputs\*.png" (
    for %%F in ("..\outputs\*.png") do (
        findstr /m /c:"%%~nF" *.tex >nul 2>&1 && (
            copy /y "%%~F" ".\" >nul
            echo   copied %%~nxF
        )
    )
)

REM Date stamp
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value 2^>nul') do set "dt=%%a"
if not defined dt for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "dt=%%a"
set "datestr=!dt:~0,8!"

REM Build main manuscript
echo.
echo [1/1] Building main manuscript ...
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
bibtex main >nul 2>&1
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
pdflatex -interaction=nonstopmode main.tex >nul 2>&1

if not exist main.pdf (
    echo ERROR: main.pdf was not generated. Check main.log for errors.
    exit /b 1
)

set "mainout=Paper1_Manuscript_FRL_!datestr!.pdf"
copy /y main.pdf "!mainout!" >nul
echo   Generated: !mainout!
for %%E in (aux log out bbl blg spl toc lof lot nav snm vrb synctex.gz fls fdb_latexmk) do del main.%%E 2>nul
del main.pdf 2>nul

echo.
echo ============================================================
echo Done.
echo   Main manuscript: !mainout!
echo ============================================================
endlocal

pause
