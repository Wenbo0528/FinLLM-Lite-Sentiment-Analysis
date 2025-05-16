@echo off
setlocal enabledelayedexpansion

:: 设置默认参数
set MODEL_PATH=FinLLM-Instruction-tuning\model_lora
set QUERY_FILE=FinLLM-RAG\data\queries.txt
set OUTPUT_DIR=FinLLM-RAG\results
set TOP_K=3

:: 解析命令行参数
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--model_path" (
    set MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--query_file" (
    set QUERY_FILE=%~2
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
if "%~1"=="--top_k" (
    set TOP_K=%~2
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
if not exist "FinLLM-RAG\inference" (
    echo Error: RAG inference directory not found
    exit /b 1
)

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "logs" mkdir logs

:: 开始RAG推理
echo [%date% %time%] Starting RAG inference...
echo [%date% %time%] Model path: %MODEL_PATH%
echo [%date% %time%] Query file: %QUERY_FILE%
echo [%date% %time%] Output directory: %OUTPUT_DIR%
echo [%date% %time%] Top-K: %TOP_K%

python FinLLM-RAG/inference/rag_retrieve_and_infer.py ^
    --model_path "%MODEL_PATH%" ^
    --query_file "%QUERY_FILE%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --top_k "%TOP_K%" ^
    > "logs\rag_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log" 2>&1

if %ERRORLEVEL% equ 0 (
    echo [%date% %time%] RAG inference completed successfully!
) else (
    echo [%date% %time%] Error: RAG inference failed
    exit /b 1
)

echo RAG inference completed!
pause 