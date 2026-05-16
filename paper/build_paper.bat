@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================
echo Building Paper 1: Main Manuscript + Internet Appendix
echo ============================================================

REM -----------------------------------------------------------
REM Archive: move any existing Paper1_*.pdf or
REM Regime_Labels_Are_Not_Representation_Invariant_*.pdf in this
REM directory into history\ so the working dir contains only the
REM freshly built artefacts.
REM -----------------------------------------------------------
if not exist "history" mkdir "history"
set "archived=0"
for %%F in (Paper1_*.pdf Regime_Labels_Are_Not_Representation_Invariant_*.pdf) do (
    if exist "%%~F" (
        move /y "%%~F" "history\" >nul 2>&1
        echo   archived %%~nxF -^> history\
        set /a archived+=1
    )
)
if !archived! gtr 0 (
    echo Archived !archived! historical PDF^(s^) to history\
)

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

REM -----------------------------------------------------------
REM Stage 1/2: Clean manuscript + Internet Appendix
REM   xr-hyper requires both .aux files; we therefore alternate
REM   pdflatex passes between main and supplement so each picks up
REM   the other's labels.
REM -----------------------------------------------------------
echo.
echo [1/2] Building main manuscript + Internet Appendix ...

REM First pass: produce both .aux files (cross-refs unresolved).
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
pdflatex -interaction=nonstopmode supplement.tex >nul 2>&1

REM bibtex on both.
bibtex main >nul 2>&1
bibtex supplement >nul 2>&1

REM Second pass: resolve cross-doc refs and bib citations.
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
pdflatex -interaction=nonstopmode supplement.tex >nul 2>&1

REM Third pass: finalise.
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
pdflatex -interaction=nonstopmode supplement.tex >nul 2>&1

if not exist main.pdf (
    echo   ERROR: main.pdf was not generated. Check main.log for errors.
    exit /b 1
)
if not exist supplement.pdf (
    echo   ERROR: supplement.pdf was not generated. Check supplement.log for errors.
    exit /b 1
)

set "mainout=Paper1_Manuscript_FRL_!datestr!.pdf"
set "supplout=Paper1_Appendix_FRL_!datestr!.pdf"
copy /y main.pdf "!mainout!" >nul
copy /y supplement.pdf "!supplout!" >nul
echo   Generated: !mainout!
echo   Generated: !supplout!
for %%E in (aux log out bbl blg spl toc lof lot nav snm vrb synctex.gz fls fdb_latexmk) do (
    del main.%%E 2>nul
    del supplement.%%E 2>nul
)
del main.pdf 2>nul
del supplement.pdf 2>nul

REM -----------------------------------------------------------
REM Stage 2/2: Marked-up diff against the latest submitted baseline
REM -----------------------------------------------------------
REM
REM Baseline = the highest-numbered main_submitted_vN.tex in this folder.
REM
REM To archive a new submission:
REM     copy main.tex main_submitted_vN.tex     :: increment N each round
REM
REM The next build automatically diffs against the latest vN. Older vN
REM files stay as version history (tracked by git).
REM
REM To skip the diff stage: pass  NO_DIFF=1  as env var.
REM -----------------------------------------------------------

if defined NO_DIFF (
    echo.
    echo [2/2] Skipping diff stage ^(NO_DIFF set^).
    goto :DONE
)

echo.
echo [2/2] Building marked-up diff against latest submitted baseline ...

REM Step 2a: locate latest main_submitted_vN.tex (numeric sort by N)
set "baseline="
for /f "delims=" %%F in ('powershell -NoProfile -Command "Get-ChildItem main_submitted_v*.tex -ErrorAction SilentlyContinue | Where-Object { $_.BaseName -match '^main_submitted_v\d+$' } | Sort-Object { [int]($_.BaseName -replace 'main_submitted_v','') } | Select-Object -Last 1 -ExpandProperty Name"') do set "baseline=%%F"

if not defined baseline (
    echo   WARNING: No main_submitted_v*.tex baseline found. Skipping diff stage.
    echo   ^(To create one: copy main.tex main_submitted_v1.tex^)
    goto :DONE
)
echo   Using baseline: !baseline!

REM Step 2b: check latexdiff-fast availability
where latexdiff-fast >nul 2>&1
if errorlevel 1 (
    echo   WARNING: latexdiff-fast not on PATH. Skipping diff stage.
    goto :DONE
)

REM Step 2c: generate diff
latexdiff-fast --math-markup=0 --graphics-markup=0 --append-safecmd="cmidrule,cline,addlinespace" "!baseline!" main.tex > main_diff.tex 2>nul
if not exist main_diff.tex (
    echo   WARNING: latexdiff-fast produced no output. Skipping.
    goto :DONE
)

REM Step 2d: compile
pdflatex -interaction=nonstopmode main_diff.tex >nul 2>&1
bibtex main_diff >nul 2>&1
pdflatex -interaction=nonstopmode main_diff.tex >nul 2>&1
pdflatex -interaction=nonstopmode main_diff.tex >nul 2>&1

if not exist main_diff.pdf (
    echo   WARNING: main_diff.pdf was not generated. Check main_diff.log for errors.
    echo   ^(Diff stage failed; clean manuscript built successfully above.^)
    goto :DONE
)

set "diffout=Paper1_Manuscript_FRL_!datestr!_marked.pdf"
copy /y main_diff.pdf "!diffout!" >nul
echo   Generated: !diffout!  ^(red = removed; blue/underline = added^)
for %%E in (aux log out bbl blg spl toc lof lot nav snm vrb synctex.gz fls fdb_latexmk tex) do del main_diff.%%E 2>nul
del main_diff.pdf 2>nul

:DONE
echo.
echo ============================================================
echo Done.
echo   Main manuscript    : !mainout!
echo   Internet Appendix  : !supplout!
if exist "Paper1_Manuscript_FRL_!datestr!_marked.pdf" echo   Marked diff        : Paper1_Manuscript_FRL_!datestr!_marked.pdf
echo ============================================================
endlocal

pause
