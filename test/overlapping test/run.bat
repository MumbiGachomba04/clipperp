@echo off
setlocal enabledelayedexpansion

set EXEC=demo_clique.exe
set LOG_FILE=run_results.log
set NP_LIST=1 2 4 8 12

echo Run started at %DATE% %TIME% > %LOG_FILE%
echo ============================== >> %LOG_FILE%

for %%G in (*.txt) do (
    echo -------- Graph: %%~nG.txt -------- >> %LOG_FILE%
    
    for %%N in (%NP_LIST%) do (
        echo Running with %%N processes on %%~nG.txt
        echo ----- mpiexec -n %%N %EXEC% %%G ----- >> %LOG_FILE%
        echo Time: %DATE% %TIME% >> %LOG_FILE%
        
        mpiexec -n %%N %EXEC% %%G >> %LOG_FILE% 2>&1

        echo ----------------------------- >> %LOG_FILE%
    )
    echo ======================================== >> %LOG_FILE%
)

echo Done. Results: %LOG_FILE%
pause