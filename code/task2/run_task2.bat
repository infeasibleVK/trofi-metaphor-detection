@echo off
setlocal

if "%~1"=="" (
  echo Usage: run_task2.bat ^<glove_path^> ^<elmo_options^> ^<elmo_weights^>
  exit /b 2
)
if "%~2"=="" (
  echo Usage: run_task2.bat ^<glove_path^> ^<elmo_options^> ^<elmo_weights^>
  exit /b 2
)
if "%~3"=="" (
  echo Usage: run_task2.bat ^<glove_path^> ^<elmo_options^> ^<elmo_weights^>
  exit /b 2
)

set ROOT=%~dp0..\..
python "%ROOT%\code\task2\train.py" ^
  --train_path "%ROOT%\data\trofi_train.csv" ^
  --val_path "%ROOT%\data\trofi_val.csv" ^
  --glove_path "%~1" ^
  --elmo_options "%~2" ^
  --elmo_weights "%~3" ^
  --save_dir "%ROOT%\checkpoints\task2"
if errorlevel 1 exit /b %errorlevel%

python "%ROOT%\code\task2\evaluate_task2.py" ^
  --test_path "%ROOT%\data\trofi_test.csv" ^
  --model_path "%ROOT%\checkpoints\task2\best_model.pt" ^
  --glove_path "%~1" ^
  --elmo_options "%~2" ^
  --elmo_weights "%~3"
endlocal
