@echo off
setlocal enabledelayedexpansion

:: Set default parameters
set MODEL_PATH=FinLLM-Instruction-tuning\model_lora
set QUERY_FILE=FinLLM-Instruction-tuning\data\validation_data.jsonl
set KNOWLEDGE_BASE=FinLLM-RAG\data\phrasebank_75_agree.json
set OUTPUT_DIR=FinLLM-RAG\results
set TOP_K=3

:: Parse command line arguments
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
if "%~1"=="--knowledge_base" (
    set KNOWLEDGE_BASE=%~2
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

:: Check Python environment
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed
    exit /b 1
)

:: Check required directories
if not exist "FinLLM-RAG\inference" (
    echo Error: RAG inference directory not found
    exit /b 1
)

:: Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "logs" mkdir logs

:: Start RAG inference
echo [%date% %time%] Starting RAG inference...
echo [%date% %time%] Model path: %MODEL_PATH%
echo [%date% %time%] Query file: %QUERY_FILE%
echo [%date% %time%] Knowledge base: %KNOWLEDGE_BASE%
echo [%date% %time%] Output directory: %OUTPUT_DIR%
echo [%date% %time%] Top-K: %TOP_K%

python FinLLM-RAG/inference/rag_retrieve_and_infer.py ^
    --model_path "%MODEL_PATH%" ^
    --query_file "%QUERY_FILE%" ^
    --knowledge_base "%KNOWLEDGE_BASE%" ^
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