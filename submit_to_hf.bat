@echo off
echo ============================================
echo   HUGGING FACE SPACE: FINAL FIXED SUBMITTER
echo ============================================
echo.
rmdir /s /q .git
git init
git add .
git commit -m "Official Hackathon Submission (Main branch sync)"
git branch -M main
git remote add huggingface https://huggingface.co/spaces/HimanshuAlien/Hakcathon.git
echo.
echo [ATTENTION] In the next step, Git will ask for your Hugging Face credentials.
echo For the password, use your Hugging Face ACCESS TOKEN (hf_...).
echo.
git push -u huggingface main --force
echo.
echo ============================================
echo   DONE! Check https://huggingface.co/spaces/HimanshuAlien/Hakcathon
echo ============================================
pause
