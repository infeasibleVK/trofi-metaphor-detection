@echo off
setlocal
if "%~1"=="" (
  echo Usage: run_task3.bat ^<glove_path^> ^<elmo_options^> ^<elmo_weights^>
  exit /b 2
)
if "%~2"=="" exit /b 2
if "%~3"=="" exit /b 2
set ROOT=%~dp0..\..
python "%ROOT%\code\task3\correlation.py"
if errorlevel 1 exit /b %errorlevel%
python "%ROOT%\code\task3\augmented_model.py" ^
  --glove_path "%~1" ^
  --elmo_options "%~2" ^
  --elmo_weights "%~3"
if errorlevel 1 exit /b %errorlevel%
python "%ROOT%\code\task3\disagreement_analysis.py"
if errorlevel 1 exit /b %errorlevel%
python "%ROOT%\code\task3\evaluate_task3.py"
endlocal
