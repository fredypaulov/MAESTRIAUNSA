#!/bin/bash
echo "================================================"
echo " INSTALANDO SISTEMA ACADEMICO MINEDU V5.1"
echo " Desarrollado por Alan Turing"
echo "================================================"
echo ""

echo "[1/3] Actualizando pip..."
python3 -m pip install --upgrade pip

echo ""
echo "[2/3] Instalando dependencias principales..."
pip3 install streamlit pandas numpy plotly scikit-learn openpyxl xlsxwriter

echo ""
echo "[3/3] Instalando dependencias opcionales..."
pip3 install xgboost catboost reportlab python-docx

echo ""
echo "================================================"
echo " INSTALACION COMPLETADA!"
echo "================================================"
echo ""
echo "Para ejecutar el sistema:"
echo "  streamlit run app_dashboard.py"
