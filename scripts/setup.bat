echo ==^> Creating virtual environment .\venv
python -m venv venv

echo ==^> Activating venv
call venv\Scripts\activate.bat

echo ==^> Upgrading pip
python -m pip install --upgrade pip

echo ==^> Installing requirements
pip install -r requirements.txt

echo ==^> Setting up .env
if not exist config\.env (
    copy config\.env.example config\.env
    echo     Created config\.env (please add your ANTHROPIC_API_KEY^)
)

echo ==^> Done.
echo.
echo Next steps:
echo   1. Edit config\.env and set ANTHROPIC_API_KEY=your-key
echo   2. python main.py --step all
echo   3. streamlit run app\streamlit_app.py
