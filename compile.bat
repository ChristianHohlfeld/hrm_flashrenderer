@echo off
setlocal

:: Find .tex file in current directory
set TEX_FILE=
for %%f in (*.tex) do (
    set TEX_FILE=%%f
)

if "%TEX_FILE%"=="" (
    echo [FEHLER] Keine .tex Datei gefunden!
    pause
    exit /b 1
)

echo Kompiliere: %TEX_FILE%
echo.

:: Run pdflatex twice (for references/TOC to update)
pdflatex -interaction=nonstopmode "%TEX_FILE%"
if errorlevel 1 goto error

echo.
echo Zweiter Durchlauf (fuer Referenzen)...
pdflatex -interaction=nonstopmode "%TEX_FILE%"
if errorlevel 1 goto error

echo.
echo [OK] PDF erstellt: %TEX_FILE:.tex=.pdf%

:: Open the PDF automatically
start "" "%TEX_FILE:.tex=.pdf%"
goto end

:error
echo.
echo [FEHLER] Kompilierung fehlgeschlagen! Siehe .log Datei fuer Details.

:end
pause
