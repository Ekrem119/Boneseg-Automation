@echo off
:: 1. Activate the environment
call env\Scripts\activate.bat

:: 2. Run the PCA script (make sure the filename matches exactly)
python pca.py

pause