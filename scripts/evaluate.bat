@echo off
setlocal enabledelayedexpansion

:: 设置默认参数
set PREDICTIONS_DIR=FinLLM-RAG\results
set GROUND_TRUTH_FILE=FinLLM-RAG\data\ground_truth.json
set METRICS_FILE=FinLLM-RAG\results\metrics.json
set EVAL_TYPE=all

:: 解析命令行参数
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--predictions_dir" (
    set PREDICTIONS_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--ground_truth_file" (
    set GROUND_TRUTH_FILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--metrics_file" (
    set METRICS_FILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--eval_type" (
    set EVAL_TYPE=%~2
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
    echo Error: Evaluation directory not found
    exit /b 1
)

:: 创建输出目录
for %%F in ("%METRICS_FILE%") do set "METRICS_DIR=%%~dpF"
if not exist "%METRICS_DIR%" mkdir "%METRICS_DIR%"
if not exist "logs" mkdir logs

:: 开始评估
echo [%date% %time%] Starting evaluation process...
echo [%date% %time%] Predictions directory: %PREDICTIONS_DIR%
echo [%date% %time%] Ground truth file: %GROUND_TRUTH_FILE%
echo [%date% %time%] Metrics file: %METRICS_FILE%
echo [%date% %time%] Evaluation type: %EVAL_TYPE%

python FinLLM-RAG/inference/evaluate_rag_results.py ^
    --predictions_dir "%PREDICTIONS_DIR%" ^
    --ground_truth_file "%GROUND_TRUTH_FILE%" ^
    --metrics_file "%METRICS_FILE%" ^
    --eval_type "%EVAL_TYPE%" ^
    > "logs\evaluate_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log" 2>&1

if %ERRORLEVEL% equ 0 (
    echo [%date% %time%] Evaluation completed successfully!
    echo [%date% %time%] Results saved to: %METRICS_FILE%
) else (
    echo [%date% %time%] Error: Evaluation failed
    exit /b 1
) 