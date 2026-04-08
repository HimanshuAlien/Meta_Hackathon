@echo off
echo ============================================
echo   OPENENV HACKATHON: AUTOMATIC SUBMITTER
echo ============================================
echo.
git init
git add .
git commit -m "Official Hackathon Submission"
git remote add origin https://github.com/HimanshuAlien/Meta_Hackathon.git
echo.
echo [ATTENTION] In the next step, Git will ask for your credentials.
echo If you use 2FA, please use your GitHub Personal Access Token (PAT).
echo.
git push -u origin main --force
echo.
echo ============================================
echo   DONE! Check https://github.com/HimanshuAlien/Meta_Hackathon
echo ============================================
pause
