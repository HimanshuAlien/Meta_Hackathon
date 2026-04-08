@echo off
echo ============================================
echo   FORCE RESET & PUSH - OPENENV SUBMIT
echo ============================================
echo.
rmdir /s /q .git
git init
git add .
git commit -m "Official Hackathon Submission (Clean Refresh)"
git remote add origin https://github.com/HimanshuAlien/Meta_Hackathon.git
echo.
echo [FINAL PUSH] Enter your credentials when prompted:
echo.
git push -u origin main --force
echo.
echo ============================================
echo   DONE! Check https://github.com/HimanshuAlien/Meta_Hackathon
echo ============================================
pause
