# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     FUNCIONES AUXILIARES GENÉRICAS                        ║
║          Utilidades para manejo de datos, búsqueda, conversiones         ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import io

def find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    Encuentra la primera columna que coincida con alguna keyword
    
    Args:
        df: DataFrame
        keywords: Lista de palabras clave a buscar
        
    Returns:
        Nombre de la columna encontrada o None
    """
    for col in df.columns:
        col_upper = str(col).upper()
        if any(kw.upper() in col_upper for kw in keywords):
            return col
    return None

def safe_slice(lista: List, start: int, end: int) -> List:
    """
    Extrae una sublista de forma segura sin errores de índice
    
    Args:
        lista: Lista original
        start: Índice inicial
        end: Índice final
        
    Returns:
        Sublista extraída
    """
    try:
        return lista[start:end] if len(lista) > start else []
    except:
        return []

def limpiar_nombre_columna(nombre: str) -> str:
    """
    Limpia y normaliza nombres de columnas
    
    Args:
        nombre: Nombre de columna original
        
    Returns:
        Nombre limpio
    """
    return str(nombre).strip().replace('\n', ' ').replace('  ', ' ')

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Datos") -> bytes:
    """
    Convierte DataFrame a bytes de Excel para descarga
    
    Args:
        df: DataFrame a convertir
        sheet_name: Nombre de la hoja
        
    Returns:
        Bytes del archivo Excel
    """
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            
            # Ajustar anchos de columna
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(idx, idx, min(max_len, 50))
        
        return output.getvalue()
    except Exception as e:
        print(f"Error al generar Excel: {e}")
        return b""

def detectar_fila_encabezado(df_raw: pd.DataFrame, keywords: List[str]) -> Optional[int]:
    """
    Detecta automáticamente la fila de encabezado en un DataFrame
    
    Args:
        df_raw: DataFrame sin procesar
        keywords: Palabras clave que indican el encabezado
        
    Returns:
        Índice de la fila de encabezado o None
    """
    for i in range(min(15, len(df_raw))):
        try:
            fila_str = ' '.join(str(x).upper() for x in df_raw.iloc[i] if pd.notna(x))
            if any(kw in fila_str for kw in keywords):
                return i
        except:
            continue
    return 0

def calcular_porcentaje_seguro(parte: float, total: float) -> float:
    """
    Calcula porcentaje de forma segura evitando división por cero
    
    Args:
        parte: Valor parcial
        total: Valor total
        
    Returns:
        Porcentaje calculado
    """
    return round((parte / total * 100), 2) if total > 0 else 0.0
