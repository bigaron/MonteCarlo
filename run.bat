cmake -G "Ninja" . -B ./build
if %errorLevel% neq 0 exit /b %errorLevel%
cd ./build
ninja
start ./MonteCarlo