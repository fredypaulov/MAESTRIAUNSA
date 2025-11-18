# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     VISTA DIRECTOR - ANÁLISIS GLOBAL                      ║
║              Análisis consolidado de toda la institución                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

from constantes import INFO_INSTITUCION, ESCALA_CALIFICACIONES
from procesamiento import obtener_columnas_notas, procesar_datos
from visualizaciones import (
    mostrar_kpis,
    crear_grafico_pastel_niveles,
    crear_mapa_calor_aulas,
    crear_grafico_comparativo_aulas,
    generar_tabla_frecuencias
)
from utils import find_column, df_to_excel_bytes, calcular_porcentaje_seguro

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL: VISTA DIRECTOR
# ═════════════════════════════════════════════════════════════════════════════

def pagina_vista_director(datos_cargados: Dict, datos_raw: Dict):
    """
    👨‍🏫 VISTA DIRECTOR: Análisis Global Institucional Completo
    
    Funcionalidades:
    - Consolidación de datos de todas las aulas
    - Métricas KPI institucionales
    - Tabla de frecuencias ponderado global
    - Identificación de mejores alumnos y estudiantes en riesgo
    - Mapas de calor y análisis comparativos
    - Recomendaciones pedagógicas
    - Exportación de reportes
    """
    
    # Encabezado
    st.title("👨‍🏫 Vista Director: Análisis Global Institucional")
    st.caption(f"📍 {INFO_INSTITUCION['nombre_ie1']} | {INFO_INSTITUCION['ubicacion']}")
    
    if not datos_cargados:
        st.warning("⚠️ No hay datos cargados. Por favor, cargue un archivo Excel.")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. CONSOLIDACIÓN DE DATOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    df_consolidado_list = []
    resumen_por_aula = []
    errores = []
    
    with st.spinner("📊 Consolidando datos de todas las aulas..."):
        for nombre_hoja, df_hoja in datos_cargados.items():
            try:
                columnas_notas, columnas_id = obtener_columnas_notas(df_hoja)
                
                if not columnas_notas:
                    errores.append(f"Hoja '{nombre_hoja}': Sin columnas de notas")
                    continue
                
                df_procesado, _ = procesar_datos(df_hoja, columnas_notas)
                df_procesado['AULA'] = nombre_hoja
                df_consolidado_list.append(df_procesado)
                
                # Métricas por aula
                total_est = len(df_procesado)
                promedio_aula = df_procesado['PROMEDIO'].mean()
                aprobados = (df_procesado['ESTADO'] == 'Aprobado').sum()
                tasa_aprob = calcular_porcentaje_seguro(aprobados, total_est)
                
                resumen_por_aula.append({
                    'AULA': nombre_hoja,
                    'ESTUDIANTES': total_est,
                    'PROMEDIO': round(promedio_aula, 2),
                    'APROBADOS': aprobados,
                    'DESAPROBADOS': total_est - aprobados,
                    'TASA_APROBACION': tasa_aprob
                })
                
            except Exception as e:
                errores.append(f"Hoja '{nombre_hoja}': {str(e)}")
                continue
    
    if errores:
        with st.expander("⚠️ Ver errores de procesamiento"):
            for error in errores:
                st.warning(error)
    
    if not df_consolidado_list:
        st.error("❌ No se pudieron procesar datos de ninguna hoja.")
        return
    
    df_global = pd.concat(df_consolidado_list, ignore_index=True)
    df_resumen_aulas = pd.DataFrame(resumen_por_aula)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. KPIs PRINCIPALES
    # ═══════════════════════════════════════════════════════════════════════════
    
    total_estudiantes = len(df_global)
    promedio_general = df_global['PROMEDIO'].mean()
    total_aprobados = (df_global['ESTADO'] == 'Aprobado').sum()
    tasa_aprobacion = calcular_porcentaje_seguro(total_aprobados, total_estudiantes)
    
    st.markdown("### 📊 Métricas Institucionales Consolidadas")
    mostrar_kpis(total_estudiantes, promedio_general, tasa_aprobacion)
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. TABLA DE FRECUENCIAS PONDERADO GLOBAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📈 Tabla de Frecuencias Ponderado Global")
    
    df_frecuencias = generar_tabla_frecuencias(df_global)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            df_frecuencias[['NIVEL', 'DESCRIPCIÓN', 'ESTUDIANTES', 'PORCENTAJE']],
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        fig_pastel = crear_grafico_pastel_niveles(df_frecuencias, 'Distribución de Niveles de Logro')
        st.plotly_chart(fig_pastel, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. MEJORES ESTUDIANTES Y ESTUDIANTES EN RIESGO
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🏆 Mejores Estudiantes y 🚨 Estudiantes en Riesgo Académico")
    
    col_nombre = find_column(df_global, ['APELLIDOS', 'NOMBRES', 'ESTUDIANTE'])
    
    col_mejores, col_riesgo = st.columns(2)
    
    with col_mejores:
        st.markdown("#### 🏆 Top 10 Mejores Estudiantes")
        
        if col_nombre:
            df_mejores = df_global.nlargest(10, 'PROMEDIO')[[col_nombre, 'AULA', 'PROMEDIO', 'CALIFICACION_LETRA']]
            df_mejores = df_mejores.reset_index(drop=True)
            df_mejores.index += 1
            st.dataframe(df_mejores, use_container_width=True)
        else:
            st.warning("No se pudo identificar columna de nombres")
    
    with col_riesgo:
        st.markdown("#### 🚨 Estudiantes en Riesgo (C)")
        
        df_riesgo = df_global[df_global['CALIFICACION_LETRA'] == 'C']
        total_riesgo = len(df_riesgo)
        pct_riesgo = calcular_porcentaje_seguro(total_riesgo, total_estudiantes)
        
        st.metric("Total en Nivel C", total_riesgo, delta=f"{pct_riesgo:.1f}%", delta_color="inverse")
        
        if total_riesgo > 0 and col_nombre:
            df_riesgo_top = df_riesgo.nsmallest(10, 'PROMEDIO')[[col_nombre, 'AULA', 'PROMEDIO']]
            df_riesgo_top = df_riesgo_top.reset_index(drop=True)
            df_riesgo_top.index += 1
            st.dataframe(df_riesgo_top, use_container_width=True)
        else:
            st.success("✅ No hay estudiantes en nivel C")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. MAPA DE CALOR POR AULA
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🗺️ Mapa de Calor: Desempeño por Aula")
    
    df_pivot = df_resumen_aulas.set_index('AULA')[['PROMEDIO', 'TASA_APROBACION']]
    fig_heatmap = crear_mapa_calor_aulas(df_pivot)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6. TABLA RESUMEN POR AULA
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📋 Resumen Detallado por Aula")
    
    st.dataframe(
        df_resumen_aulas,
        use_container_width=True,
        hide_index=True,
        column_config={
            'AULA': st.column_config.TextColumn('Aula', width='medium'),
            'ESTUDIANTES': st.column_config.NumberColumn('Total', format='%d'),
            'PROMEDIO': st.column_config.NumberColumn('Promedio', format='%.2f'),
            'APROBADOS': st.column_config.NumberColumn('Aprobados', format='%d'),
            'DESAPROBADOS': st.column_config.NumberColumn('Desaprobados', format='%d'),
            'TASA_APROBACION': st.column_config.NumberColumn('Tasa Aprob.', format='%.1f%%')
        }
    )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 7. LISTA COMPLETA DE ESTUDIANTES
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📝 Lista Completa de Estudiantes")
    
    if col_nombre:
        df_lista = df_global[[col_nombre, 'AULA', 'PROMEDIO', 'CALIFICACION_LETRA', 'ESTADO']].copy()
        df_lista = df_lista.sort_values(['AULA', 'PROMEDIO'], ascending=[True, False])
        df_lista = df_lista.reset_index(drop=True)
        df_lista.index += 1
        
        st.dataframe(df_lista, use_container_width=True, height=400)
        
        csv = df_lista.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Descargar Lista Completa (CSV)",
            data=csv,
            file_name=f"lista_estudiantes_global_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 8. GRÁFICOS COMPARATIVOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📊 Análisis Comparativo por Aula")
    
    fig1, fig2 = crear_grafico_comparativo_aulas(df_resumen_aulas)
    
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.plotly_chart(fig1, use_container_width=True)
    with col_graf2:
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 9. RECOMENDACIONES PEDAGÓGICAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 💡 Recomendaciones Pedagógicas Institucionales")
    
    pct_ad = calcular_porcentaje_seguro((df_global['CALIFICACION_LETRA'] == 'AD').sum(), total_estudiantes)
    pct_c = calcular_porcentaje_seguro((df_global['CALIFICACION_LETRA'] == 'C').sum(), total_estudiantes)
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    
    with col_rec1:
        if tasa_aprobacion >= 80:
            st.success(f"""
            ✅ **Excelente Desempeño Institucional**
            
            Tasa de aprobación: {tasa_aprobacion:.1f}%
            
            **Acciones sugeridas:**
            - Mantener estrategias pedagógicas actuales
            - Compartir buenas prácticas entre docentes
            - Implementar programas de mentoría estudiante-estudiante
            """)
        elif tasa_aprobacion >= 60:
            st.warning(f"""
            ⚠️ **Desempeño Aceptable con Áreas de Mejora**
            
            Tasa de aprobación: {tasa_aprobacion:.1f}%
            
            **Acciones sugeridas:**
            - Reforzar acompañamiento pedagógico
            - Implementar círculos de estudio
            - Capacitación docente en evaluación formativa
            """)
        else:
            st.error(f"""
            🚨 **Requiere Intervención Urgente**
            
            Tasa de aprobación: {tasa_aprobacion:.1f}%
            
            **Acciones sugeridas:**
            - Plan de recuperación pedagógica inmediato
            - Reunión con padres de familia
            - Coordinación con UGEL para soporte adicional
            """)
    
    with col_rec2:
        if pct_ad >= 15:
            st.info(f"""
            🌟 **Alto Porcentaje de Logro Destacado**
            
            {pct_ad:.1f}% en nivel AD
            
            **Oportunidades:**
            - Programa de estudiantes destacados
            - Proyectos de investigación escolar
            - Preparación para concursos académicos
            """)
        else:
            st.info(f"""
            📈 **Oportunidad de Potenciar Talentos**
            
            {pct_ad:.1f}% en nivel AD
            
            **Sugerencias:**
            - Identificar estudiantes con potencial
            - Actividades de desafío cognitivo
            - Mentoría de docentes especializados
            """)
    
    with col_rec3:
        if pct_c > 20:
            st.warning(f"""
            ⚠️ **Alto Porcentaje en Nivel C**
            
            {pct_c:.1f}% requiere reforzamiento
            
            **Plan de acción:**
            - Tutorías personalizadas
            - Material didáctico adaptado
            - Seguimiento semanal de progreso
            - Reuniones con padres/apoderados
            """)
        else:
            st.success(f"""
            ✅ **Bajo Porcentaje en Riesgo**
            
            Solo {pct_c:.1f}% en nivel C
            
            **Mantener:**
            - Estrategias preventivas actuales
            - Detección temprana de dificultades
            - Acompañamiento continuo
            """)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 10. EXPORTACIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📥 Exportar Reportes Institucionales")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if col_nombre:
            df_export = df_global[[col_nombre, 'AULA', 'PROMEDIO', 'CALIFICACION_LETRA', 'ESTADO']]
            excel_global = df_to_excel_bytes(df_export, "Reporte_Global")
            
            st.download_button(
                label="📊 Reporte Global Excel",
                data=excel_global,
                file_name=f"reporte_global_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with col_exp2:
        excel_resumen = df_to_excel_bytes(df_resumen_aulas, "Resumen_Aulas")
        st.download_button(
            label="📋 Resumen por Aula Excel",
            data=excel_resumen,
            file_name=f"resumen_aulas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_exp3:
        excel_freq = df_to_excel_bytes(df_frecuencias, "Frecuencias")
        st.download_button(
            label="📈 Tabla Frecuencias Excel",
            data=excel_freq,
            file_name=f"tabla_frecuencias_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.success("✅ Vista Director cargada correctamente")
