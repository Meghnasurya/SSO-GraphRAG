@echo off
echo Starting RAG Test Generator Application
echo =====================================
echo.

echo Starting Backend Server (Port 8000)...
echo.
start "Backend Server" cmd /k "cd /d %~dp0 && python main.py"

timeout /t 3 /nobreak > nul

echo Starting Frontend Server (Port 3000)...
echo.
start "Frontend Server" cmd /k "cd /d %~dp0frontend && python serve.py"

echo.
echo ✅ Both servers are starting...
echo 🔗 Frontend: http://localhost:3000
echo 🔗 Backend:  http://localhost:8000
echo.
echo Press any key to exit this window...
pause > nul
