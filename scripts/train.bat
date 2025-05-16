@echo off
setlocal enabledelayedexpansion

:: 设置默认参数
set MODEL_NAME=default_model
set BATCH_SIZE=8
set LEARNING_RATE=2e-5
set EPOCHS=3

:: 解析命令行参数
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--model_name" (
    set MODEL_NAME=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--batch_size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--learning_rate" (
    set LEARNING_RATE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto :parse_args
)
echo Unknown parameter: %~1
exit /b 1
:end_parse_args

:: 检查Python环境
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed
    exit /b 1
)

:: 检查必要的目录
if not exist "FinLLM-Instruction-tuning\train" (
    echo Error: Training directory not found
    exit /b 1
)

:: 创建日志目录
if not exist "logs" mkdir logs

:: 开始训练
echo [%date% %time%] Starting training process...
echo [%date% %time%] Model: %MODEL_NAME%
echo [%date% %time%] Batch size: %BATCH_SIZE%
echo [%date% %time%] Learning rate: %LEARNING_RATE%
echo [%date% %time%] Epochs: %EPOCHS%

python FinLLM-Instruction-tuning/train/train.py ^
    --model_name "%MODEL_NAME%" ^
    --batch_size "%BATCH_SIZE%" ^
    --learning_rate "%LEARNING_RATE%" ^
    --epochs "%EPOCHS%" ^
    > "logs\train_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log" 2>&1

if %ERRORLEVEL% equ 0 (
    echo [%date% %time%] Training completed successfully!
) else (
    echo [%date% %time%] Error: Training failed
    exit /b 1
) 