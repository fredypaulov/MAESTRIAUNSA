# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                   PROCESAMIENTO Y LIMPIEZA DE DATOS                       ║
║        Carga, limpieza, transformación y procesamiento de DataFrames      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List, Optional, Dict
from constantes import EQUIVALENCIAS_NOTAS, KEYWORDS_COLUMNAS
from utils import find_column, detectar_fila_encabezado
from contexto import gestor_evaluacion

# ═════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_excel(archivo_subido) -> Tuple[Optional[Dict], Optional[Dict], List[str], Optional[str]]:
    """
    Carga todas las hojas de un archivo Excel con manejo robusto de errores
    
    Returns:
        (datos_por_hoja, datos_raw_por_hoja, nombres_hojas, error)
    """
    try:
        if archivo_subido is None:
            return None, None, [], "No se proporcionó archivo"
        
        try:
            xls = pd.ExcelFile(archivo_subido)
        except Exception as e:
            return None, None, [], f"Error al leer archivo Excel: {str(e)}"
        
        nombres_hojas = xls.sheet_names
        
        if not nombres_hojas:
            return None, None, [], "El archivo no contiene hojas"
        
        datos_por_hoja = {}
        datos_raw_por_hoja = {}
        errores = []
        
        for hoja in nombres_hojas:
            try:
                df_raw = pd.read_excel(xls, sheet_name=hoja, header=None)
                datos_raw_por_hoja[hoja] = df_raw
                
                fila_header = detectar_fila_encabezado(
                    df_raw, 
                    KEYWORDS_COLUMNAS['estudiante']
                )
                
                if fila_header is not None:
                    df_hoja = pd.read_excel(xls, sheet_name=hoja, header=fila_header)
                    df_hoja = limpiar_dataframe(df_hoja)
                    
                    if not df_hoja.empty:
                        datos_por_hoja[hoja] = df_hoja
                    else:
                        errores.append(f"Hoja '{hoja}': Sin datos válidos")
                else:
                    errores.append(f"Hoja '{hoja}': No se encontró encabezado")
                    
            except Exception as e:
                errores.append(f"Hoja '{hoja}': {str(e)}")
                continue
        
        if not datos_por_hoja:
            error_msg = "No se pudieron cargar hojas válidas"
            if errores:
                error_msg += f". Errores: {'; '.join(errores[:3])}"
            return None, None, [], error_msg
        
        return datos_por_hoja, datos_raw_por_hoja, list(datos_por_hoja.keys()), None
    
    except Exception as e:
        return None, None, [], f"Error crítico: {str(e)}"

# ═════════════════════════════════════════════════════════════════════════════
# LIMPIEZA DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia DataFrame: elimina columnas/filas vacías y duplicados"""
    try:
        # Eliminar columnas duplicadas
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            indices = cols[cols == dup].index.tolist()
            cols.iloc[indices] = [f"{dup}.{i}" if i != 0 else dup for i in range(len(indices))]
        
        df.columns = cols
        
        # Eliminar columnas sin nombre
        df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed', na=False)]
        df = df.dropna(axis=1, how='all')
        
        # Eliminar filas vacías
        if len(df.columns) > 1:
            col_id = df.columns[1]
            df = df.dropna(subset=[col_id])
        
        return df.reset_index(drop=True)
    
    except Exception as e:
        st.warning(f"⚠️ Error al limpiar DataFrame: {e}")
        return df

# ═════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE COLUMNAS
# ═════════════════════════════════════════════════════════════════════════════

def obtener_columnas_notas(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detecta columnas de notas con manejo robusto de errores
    
    Returns:
        (columnas_notas, columnas_id)
    """
    columnas_notas = []
    columnas_id = []
    
    palabras_excluir = {
        'ESTUDIANTE', 'NOMBRE', 'APELLIDO', 'GRADO', 'SECCION',
        'CODIGO', 'DNI', 'ID', 'PROMEDIO', 'OBSERVACION', 'FECHA', 'ESTADO'
    }
    
    columnas_lista = list(df.columns)
    
    for col in columnas_lista:
        try:
            col_str = str(col).upper()
            
            # Identificar columnas de ID
            if any(kw in col_str for kw in ['ESTUDIANTE', 'NOMBRE', 'APELLIDO']):
                columnas_id.append(col)
                continue
            
            # Excluir columnas específicas
            if any(kw in col_str for kw in palabras_excluir):
                continue
            
            try:
                muestra = df[col].dropna()
                
                if len(muestra) < 3:
                    continue
                
                muestra_sample = muestra.sample(min(20, len(muestra)), random_state=42)
                
                # Verificar si es numérica (0-20)
                try:
                    muestra_num = pd.to_numeric(muestra_sample, errors='coerce').dropna()
                    
                    if len(muestra_sample) > 0:
                        proporcion_numerica = len(muestra_num) / len(muestra_sample)
                        
                        if proporcion_numerica > 0.7 and len(muestra_num) > 0:
                            if muestra_num.min() >= 0 and muestra_num.max() <= 20:
                                columnas_notas.append(col)
                                continue
                except Exception:
                    pass
                
                # Verificar si contiene letras (C/B/A/AD)
                try:
                    muestra_str = muestra_sample.astype(str).str.upper().str.strip()
                    letras_validas = muestra_str.isin(['A', 'B', 'C', 'AD'])
                    
                    if len(muestra_sample) > 0:
                        proporcion_letras = letras_validas.sum() / len(muestra_sample)
                        
                        if proporcion_letras > 0.6:
                            columnas_notas.append(col)
                except Exception:
                    pass
                    
            except (KeyError, TypeError, IndexError):
                continue
                
        except Exception:
            continue
    
    return columnas_notas, columnas_id

# ═════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE NOTAS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def procesar_datos(df: pd.DataFrame, columnas_notas: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Procesa notas: convierte a numérico y calcula promedios
    
    Returns:
        (df_procesado, columnas_numericas_procesadas)
    """
    try:
        df_proc = df.copy()
        columnas_num_proc = []
        
        for col in columnas_notas:
            col_num = f"{col}_num"
            
            # Intentar conversión directa a numérico
            df_proc[col_num] = pd.to_numeric(df_proc[col], errors='coerce')
            
            # Convertir letras a números
            mask_nan = df_proc[col_num].isna()
            if mask_nan.any():
                df_proc.loc[mask_nan, col_num] = (
                    df_proc.loc[mask_nan, col]
                    .astype(str).str.upper().str.strip()
                    .map(EQUIVALENCIAS_NOTAS)
                )
            
            # Rellenar valores faltantes con nota mínima
            df_proc[col_num] = df_proc[col_num].fillna(EQUIVALENCIAS_NOTAS['C'])
            columnas_num_proc.append(col_num)
        
        if columnas_num_proc:
            # Calcular promedio
            df_proc['PROMEDIO'] = df_proc[columnas_num_proc].mean(axis=1).round(2)
            
            # Asignar calificación en letra
            df_proc['CALIFICACION_LETRA'] = df_proc['PROMEDIO'].apply(
                gestor_evaluacion.num_a_letra
            )
            
            # Determinar estado (Aprobado/Desaprobado)
            df_proc['ESTADO'] = df_proc['CALIFICACION_LETRA'].apply(
                lambda x: 'Aprobado' if x in ['AD', 'A', 'B'] else 'Desaprobado'
            )
        
        return df_proc, columnas_num_proc
    
    except Exception as e:
        st.error(f"❌ Error al procesar datos: {e}")
        return df, []

# ═════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO POR ÁREAS
# ═════════════════════════════════════════════════════════════════════════════

def procesar_datos_por_area(df: pd.DataFrame, columnas_notas: List[str]) -> Dict[str, List[str]]:
    """
    Agrupa las notas por área curricular
    
    Returns:
        Diccionario área -> lista de columnas
    """
    try:
        from utils import safe_slice
        
        return {
            'MATEMÁTICA': safe_slice(columnas_notas, 0, 4),
            'COMUNICACIÓN': safe_slice(columnas_notas, 4, 7),
            'CIENCIA Y TECNOLOGÍA': safe_slice(columnas_notas, 7, 10),
            'CIENCIAS SOCIALES': safe_slice(columnas_notas, 10, 13),
            'DPCC': safe_slice(columnas_notas, 13, 15),
            'EPT': safe_slice(columnas_notas, 15, 16),
            'EDUCACIÓN FÍSICA': safe_slice(columnas_notas, 16, 19),
            'ARTE Y CULTURA': safe_slice(columnas_notas, 19, 21),
            'INGLÉS': safe_slice(columnas_notas, 21, 24),
            'EDUCACIÓN RELIGIOSA': safe_slice(columnas_notas, 24, 26),
        }
    except:
        return {}
