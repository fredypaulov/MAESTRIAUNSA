# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    VISUALIZACIONES Y GRÁFICOS                             ║
║         Funciones para generar gráficos con Plotly y Streamlit           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from constantes import COLORES_NIVELES, ESCALA_CALIFICACIONES
from utils import calcular_porcentaje_seguro

# ═════════════════════════════════════════════════════════════════════════════
# KPIs Y MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════

def mostrar_kpis(total: int, promedio: float, tasa_aprob: float):
    """Muestra tarjetas KPI principales"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <p style='color: white; margin: 0; font-size: 14px;'>Total Estudiantes</p>
            <h2 style='color: white; margin: 10px 0;'>{total} 🎓</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <p style='color: white; margin: 0; font-size: 14px;'>Promedio General</p>
            <h2 style='color: white; margin: 10px 0;'>{promedio:.2f} 📊</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = '#06D6A0' if tasa_aprob >= 70 else '#FFD166' if tasa_aprob >= 50 else '#FF6B6B'
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <p style='color: white; margin: 0; font-size: 14px;'>Tasa Aprobación</p>
            <h2 style='color: white; margin: 10px 0;'>{tasa_aprob:.1f}% ✅</h2>
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICOS DE DISTRIBUCIÓN
# ═════════════════════════════════════════════════════════════════════════════

def crear_grafico_pastel_niveles(df_frecuencias: pd.DataFrame, titulo: str = "Distribución de Niveles"):
    """Crea gráfico de pastel para distribución de niveles"""
    fig = px.pie(
        df_frecuencias,
        values='ESTUDIANTES',
        names='NIVEL',
        title=titulo,
        color='NIVEL',
        color_discrete_map=COLORES_NIVELES
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=14
    )
    fig.update_layout(height=400)
    return fig

def crear_grafico_barras_horizontal(df: pd.DataFrame, x_col: str, y_col: str, titulo: str, color_col: str = None):
    """Crea gráfico de barras horizontal"""
    fig = px.bar(
        df,
        y=y_col,
        x=x_col,
        orientation='h',
        title=titulo,
        color=color_col if color_col else y_col,
        color_discrete_map=COLORES_NIVELES if color_col == 'NIVEL' else None,
        text=x_col
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    return fig

def crear_grafico_barras_vertical(df: pd.DataFrame, x_col: str, y_col: str, titulo: str, color_scale: str = 'RdYlGn'):
    """Crea gráfico de barras vertical"""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=titulo,
        color=y_col,
        color_continuous_scale=color_scale,
        text=y_col
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# MAPAS DE CALOR
# ═════════════════════════════════════════════════════════════════════════════

def crear_mapa_calor_aulas(df_pivot: pd.DataFrame, titulo: str = "Mapa de Calor por Aula"):
    """Crea mapa de calor para rendimiento por aula"""
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values.T,
        x=df_pivot.index,
        y=['Promedio (0-20)', 'Tasa Aprobación (%)'],
        colorscale='RdYlGn',
        text=np.round(df_pivot.values.T, 2),
        texttemplate='%{text}',
        textfont={"size": 12, "color": "white"},
        colorbar=dict(title="Valor", thickness=15),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={'text': titulo, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Aula",
        yaxis_title="Métrica",
        height=400,
        font=dict(size=12)
    )
    return fig

def crear_mapa_calor_areas(df_pivot: pd.DataFrame, titulo: str = "Mapa de Calor por Área"):
    """Crea mapa de calor para distribución por áreas"""
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale='Viridis',
        text=df_pivot.values,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Cantidad")
    ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Nivel de Logro",
        yaxis_title="Área",
        height=max(400, len(df_pivot) * 40)
    )
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICOS DE TENDENCIAS
# ═════════════════════════════════════════════════════════════════════════════

def crear_histograma_distribucion(df: pd.DataFrame, promedio_aula: float, umbral: float = 11.0):
    """Crea histograma de distribución de promedios"""
    fig = px.histogram(
        df,
        x='PROMEDIO',
        nbins=20,
        title='Distribución de Promedios',
        labels={'PROMEDIO': 'Promedio (0-20)', 'count': 'Cantidad de Estudiantes'},
        color_discrete_sequence=['#667eea']
    )
    
    # Línea de promedio
    fig.add_vline(
        x=promedio_aula,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio: {promedio_aula:.2f}"
    )
    
    # Línea de aprobación
    fig.add_vline(
        x=umbral,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Línea Aprobación ({umbral})"
    )
    
    fig.update_layout(height=400)
    return fig

def crear_grafico_matriz_confusion(matriz: np.ndarray):
    """Crea visualización de matriz de confusión"""
    fig = go.Figure(data=go.Heatmap(
        z=matriz,
        x=['Predicho: Desaprobado', 'Predicho: Aprobado'],
        y=['Real: Desaprobado', 'Real: Aprobado'],
        text=matriz,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title='Matriz de Confusión - Clasificación Aprobado/Desaprobado',
        height=400
    )
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICOS COMPARATIVOS
# ═════════════════════════════════════════════════════════════════════════════

def crear_grafico_comparativo_aulas(df_resumen: pd.DataFrame):
    """Crea gráficos comparativos de promedio y tasa de aprobación por aula"""
    
    # Gráfico 1: Promedio por aula
    fig1 = px.bar(
        df_resumen.sort_values('PROMEDIO', ascending=False),
        x='AULA',
        y='PROMEDIO',
        title='Promedio por Aula',
        color='PROMEDIO',
        color_continuous_scale='RdYlGn',
        text='PROMEDIO',
        range_color=[0, 20]
    )
    fig1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig1.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
    
    # Gráfico 2: Tasa de aprobación
    fig2 = px.bar(
        df_resumen.sort_values('TASA_APROBACION', ascending=False),
        x='AULA',
        y='TASA_APROBACION',
        title='Tasa de Aprobación por Aula (%)',
        color='TASA_APROBACION',
        color_continuous_scale='RdYlGn',
        text='TASA_APROBACION',
        range_color=[0, 100]
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
    
    return fig1, fig2

# ═════════════════════════════════════════════════════════════════════════════
# TABLA DE FRECUENCIAS
# ═════════════════════════════════════════════════════════════════════════════

def generar_tabla_frecuencias(df: pd.DataFrame) -> pd.DataFrame:
    """Genera tabla de frecuencias por nivel de logro"""
    total = len(df)
    frecuencias = df['CALIFICACION_LETRA'].value_counts()
    
    df_frecuencias = pd.DataFrame({
        'NIVEL': frecuencias.index,
        'ESTUDIANTES': frecuencias.values,
        'PORCENTAJE': (frecuencias.values / total * 100).round(2)
    })
    
    # Ordenar por niveles MINEDU
    orden_niveles = ['AD', 'A', 'B', 'C']
    df_frecuencias['NIVEL'] = pd.Categorical(
        df_frecuencias['NIVEL'],
        categories=orden_niveles,
        ordered=True
    )
    df_frecuencias = df_frecuencias.sort_values('NIVEL').reset_index(drop=True)
    
    # Agregar descripciones
    df_frecuencias['DESCRIPCIÓN'] = df_frecuencias['NIVEL'].map(
        lambda x: ESCALA_CALIFICACIONES.get(x, {}).get('desc', '')
    )
    
    return df_frecuencias
