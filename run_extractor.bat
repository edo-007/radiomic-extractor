@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo Avvio radiomic extractor...
echo Cartella progetto: %CD%
echo Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -m extractor.main

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo Esecuzione terminata con errore: %EXIT_CODE%
) else (
    echo Esecuzione completata.
)
echo.
pause
exit /b %EXIT_CODE%
