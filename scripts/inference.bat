@echo off
setlocal enabledelayedexpansion

:: Set default parameters
set MODEL_PATH=FinLLM-Instruction-tuning\model_lora
set INPUT_FILE=FinLLM-Instruction-tuning\data\validation_data.jsonl
set OUTPUT_DIR=FinLLM-Instruction-tuning\results

:: Parse command line arguments
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

:: Check Python environment
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed
    exit /b 1
)

:: Check required directories
if not exist "FinLLM-Instruction-tuning\Inference" (
    echo Error: Inference directory not found
    exit /b 1
)

:: Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "logs" mkdir logs

:: Start inference
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