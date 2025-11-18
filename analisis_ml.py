# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                   ANÁLISIS DE MACHINE LEARNING                            ║
║        Cálculo de métricas: ROC-AUC, F1-Score, Precision, Recall         ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict

# Importación condicional de scikit-learn
try:
    from sklearn.metrics import (
        roc_auc_score, 
        f1_score, 
        precision_score, 
        recall_score, 
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE MÉTRICAS ML
# ═════════════════════════════════════════════════════════════════════════════

def calcular_metricas_ml(df: pd.DataFrame, umbral_aprobacion: float = 11.0) -> Dict[str, float]:
    """
    Calcula métricas de Machine Learning para evaluación del rendimiento académico
    
    Args:
        df: DataFrame con columna 'PROMEDIO'
        umbral_aprobacion: Nota mínima para aprobar (default: 11.0)
        
    Returns:
        Dict con métricas: roc_auc, f1_score, precision, recall
    """
    
    if not SKLEARN_AVAILABLE:
        return {}
    
    try:
        if 'PROMEDIO' not in df.columns or len(df) == 0:
            return {}
        
        # Variable binaria: 1 = Aprobado, 0 = Desaprobado
        y_true = (df['PROMEDIO'] >= umbral_aprobacion).astype(int)
        
        # Probabilidades normalizadas (0-1)
        y_scores = df['PROMEDIO'] / 20.0
        
        # Predicciones binarias
        y_pred = (df['PROMEDIO'] >= umbral_aprobacion).astype(int)
        
        # Caso especial: todos aprobados o todos desaprobados
        if len(y_true.unique()) < 2:
            valor_base = 1.0 if y_true.iloc[0] == 1 else 0.0
            return {
                'roc_auc': valor_base,
                'f1_score': valor_base,
                'precision': valor_base,
                'recall': valor_base
            }
        
        # Calcular métricas
        metricas = {
            'roc_auc': roc_auc_score(y_true, y_scores),
            'f1_score': f1_score(y_true, y_pred, zero_division=1),
            'precision': precision_score(y_true, y_pred, zero_division=1),
            'recall': recall_score(y_true, y_pred, zero_division=1)
        }
        
        return metricas
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular métricas ML: {e}")
        return {}

def calcular_matriz_confusion(df: pd.DataFrame, umbral_aprobacion: float = 11.0) -> np.ndarray:
    """
    Calcula la matriz de confusión para clasificación aprobado/desaprobado
    
    Args:
        df: DataFrame con columna 'PROMEDIO'
        umbral_aprobacion: Nota mínima para aprobar
        
    Returns:
        Matriz de confusión 2x2 como numpy array
    """
    
    if not SKLEARN_AVAILABLE:
        return np.array([[0, 0], [0, 0]])
    
    try:
        y_true = (df['PROMEDIO'] >= umbral_aprobacion).astype(int)
        y_pred = (df['PROMEDIO'] >= umbral_aprobacion).astype(int)
        return confusion_matrix(y_true, y_pred)
    except:
        return np.array([[0, 0], [0, 0]])

def interpretar_roc_auc(score: float) -> tuple:
    """
    Interpreta el valor de ROC-AUC
    
    Args:
        score: Valor de ROC-AUC (0-1)
        
    Returns:
        (nivel, color, mensaje)
    """
    if score >= 0.9:
        return ("Excelente", "success", "✅ Excelente discriminación")
    elif score >= 0.8:
        return ("Muy bueno", "info", "ℹ️ Muy buena discriminación")
    elif score >= 0.7:
        return ("Bueno", "info", "ℹ️ Buena discriminación")
    elif score >= 0.6:
        return ("Aceptable", "warning", "⚠️ Discriminación aceptable")
    else:
        return ("Mejorable", "warning", "⚠️ Discriminación mejorable")

def interpretar_f1_score(score: float) -> tuple:
    """
    Interpreta el valor de F1-Score
    
    Args:
        score: Valor de F1-Score (0-1)
        
    Returns:
        (nivel, color, mensaje)
    """
    if score >= 0.9:
        return ("Excelente", "success", "✅ Balance excelente")
    elif score >= 0.8:
        return ("Muy bueno", "info", "ℹ️ Muy buen balance")
    elif score >= 0.7:
        return ("Bueno", "info", "ℹ️ Buen balance")
    else:
        return ("Mejorable", "warning", "⚠️ Balance mejorable")
