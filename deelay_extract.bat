@echo off
setlocal

rem Usa sempre la cartella in cui si trova questo file, anche se il batch
rem viene avviato da Esplora file o da un'altra directory.
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "CONFIG_FILE=%~dp0config\config_extractor.yaml"

if not exist "%PYTHON_EXE%" (
    echo ERRORE: interprete Python non trovato:
    echo %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

if not exist "%CONFIG_FILE%" (
    echo ERRORE: configurazione non trovata:
    echo %CONFIG_FILE%
    echo.
    pause
    exit /b 1
)

echo L'estrazione verra' avviata tra 15 ore...
timeout /t 54000 /nobreak

echo.
echo Avvio radiomic extractor...
"%PYTHON_EXE%" "%~dp0cli.py" extract --config "%CONFIG_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo Estrazione terminata.
) else (
    echo Estrazione terminata con errore: %EXIT_CODE%
)
echo.
pause
exit /b %EXIT_CODE%
