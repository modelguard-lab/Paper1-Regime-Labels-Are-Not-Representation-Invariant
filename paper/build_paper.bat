@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================
echo Building Paper 1: Main Manuscript + Internet Appendix
echo ============================================================

REM -----------------------------------------------------------
REM Probe for Perl (required by latexdiff/latexdiff-fast).
REM MiKTeX's latexdiff is a Perl script; if perl.exe is not on
REM PATH the diff stage fails silently and leaves an empty
REM main_diff.tex behind. Prefer Git for Windows' bundled Perl.
REM -----------------------------------------------------------
where perl >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\Git\usr\bin\perl.exe" (
        set "PATH=C:\Program Files\Git\usr\bin;!PATH!"
    ) else if exist "C:\Strawberry\perl\bin\perl.exe" (
        set "PATH=C:\Strawberry\perl\bin;!PATH!"
    )
)

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
REM Baseline = the highest-numbered git tag matching R* (R1, R2, R3, ...).
REM
REM To archive a new submission, tag the commit:
REM     git tag R<N> -m "FRL R<N> submission baseline"
REM     git push origin R<N>
REM
REM The next build automatically diffs against the latest R-tag. Older
REM R-tags stay as immutable submission markers in git.
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

REM Step 2a: locate latest R-tag (numeric sort by N)
set "baseline_tag="
for /f "delims=" %%T in ('powershell -NoProfile -Command "git tag -l 'R*' 2>$null | Where-Object { $_ -match '^R\d+$' } | Sort-Object { [int]($_ -replace 'R','') } | Select-Object -Last 1"') do set "baseline_tag=%%T"

if not defined baseline_tag (
    echo   WARNING: No R-tag found in git. Skipping diff stage.
    echo   ^(To create one: git tag R1 -m "FRL R1 submission baseline"^)
    goto :DONE
)
echo   Using baseline: tag !baseline_tag! ^(git^)

REM Step 2b: extract baseline main.tex from tagged commit
git show !baseline_tag!:paper/main.tex > .baseline_main.tex 2>nul
if not exist .baseline_main.tex (
    echo   WARNING: git show !baseline_tag!:paper/main.tex failed. Skipping diff stage.
    goto :DONE
)

REM Step 2c: check latexdiff-fast availability
where latexdiff-fast >nul 2>&1
if errorlevel 1 (
    echo   WARNING: latexdiff-fast not on PATH. Skipping diff stage.
    del .baseline_main.tex 2>nul
    goto :DONE
)

REM Step 2d: generate diff
latexdiff-fast --math-markup=0 --graphics-markup=0 --append-safecmd="cmidrule,cline,addlinespace" .baseline_main.tex main.tex > main_diff.tex 2>nul
del .baseline_main.tex 2>nul

REM latexdiff redirect creates main_diff.tex even on failure; check
REM the file is non-empty before continuing (silent latexdiff failures
REM otherwise leave intermediates behind).
set "diffsize=0"
for %%S in (main_diff.tex) do set "diffsize=%%~zS"
if !diffsize! lss 100 (
    echo   WARNING: latexdiff-fast produced empty output ^(Perl missing or baseline error^). Skipping.
    goto :CLEANUP_DIFF
)

REM Step 2e: compile
pdflatex -interaction=nonstopmode main_diff.tex >nul 2>&1
bibtex main_diff >nul 2>&1
pdflatex -interaction=nonstopmode main_diff.tex >nul 2>&1
pdflatex -interaction=nonstopmode main_diff.tex >nul 2>&1

if not exist main_diff.pdf (
    echo   WARNING: main_diff.pdf was not generated. Check main_diff.log for errors.
    echo   ^(Diff stage failed; clean manuscript built successfully above.^)
    goto :CLEANUP_DIFF
)

set "diffout=Paper1_Manuscript_FRL_!datestr!_marked.pdf"
copy /y main_diff.pdf "!diffout!" >nul
echo   Generated: !diffout!  ^(red = removed; blue/underline = added^)

:CLEANUP_DIFF
REM Always sweep diff-stage intermediates, regardless of success/failure.
for %%E in (aux log out bbl blg spl toc lof lot nav snm vrb synctex.gz fls fdb_latexmk tex pdf) do del main_diff.%%E 2>nul
REM latexdiff-fast leaves DiffA-*/DiffB-* working files on Windows tmp drop-throughs.
del /q DiffA-* 2>nul
del /q DiffB-* 2>nul

:DONE
REM Final unconditional sweep, in case anything escaped the per-stage cleanups.
for %%E in (aux log out bbl blg spl toc lof lot nav snm vrb synctex.gz fls fdb_latexmk) do (
    del main.%%E 2>nul
    del supplement.%%E 2>nul
    del main_diff.%%E 2>nul
)
del main.pdf supplement.pdf main_diff.pdf main_diff.tex 2>nul
del /q DiffA-* DiffB-* 2>nul

echo.
echo ============================================================
echo Done.
echo   Main manuscript    : !mainout!
echo   Internet Appendix  : !supplout!
if exist "Paper1_Manuscript_FRL_!datestr!_marked.pdf" echo   Marked diff        : Paper1_Manuscript_FRL_!datestr!_marked.pdf
echo ============================================================
endlocal

pause
