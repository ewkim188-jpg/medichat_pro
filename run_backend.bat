@echo off
echo ===================================================
echo MediChat Pro - Local AI Backend Server
echo ===================================================
echo Starting FastAPI server with Ngrok tunnel for Streamlit Cloud...
python backend.py --ngrok
pause
