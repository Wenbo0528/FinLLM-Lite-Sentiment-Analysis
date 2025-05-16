@echo off
setlocal enabledelayedexpansion

:: 设置默认参数
set MODEL_PATH=FinLLM-Instruction-tuning\model_lora
set INPUT_FILE=FinLLM-Instruction-tuning\data\test_queries.txt
set OUTPUT_DIR=FinLLM-Instruction-tuning\results

:: 解析命令行参数
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--model_path" (
    set MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--input_file" (
    set INPUT_FILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output_dir" (
    set OUTPUT_DIR=%~2
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
if not exist "FinLLM-Instruction-tuning\Inference" (
    echo Error: Inference directory not found
    exit /b 1
)

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "logs" mkdir logs

:: 开始推理
echo [%date% %time%] Starting inference process...
echo [%date% %time%] Model path: %MODEL_PATH%
echo [%date% %time%] Input file: %INPUT_FILE%
echo [%date% %time%] Output directory: %OUTPUT_DIR%

python FinLLM-Instruction-tuning/Inference/inference.py ^
    --model_path "%MODEL_PATH%" ^
    --input_file "%INPUT_FILE%" ^
    --output_dir "%OUTPUT_DIR%" ^
    > "logs\inference_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log" 2>&1

if %ERRORLEVEL% equ 0 (
    echo [%date% %time%] Inference completed successfully!
    echo [%date% %time%] Results saved to: %OUTPUT_DIR%
) else (
    echo [%date% %time%] Error: Inference failed
    exit /b 1
) 