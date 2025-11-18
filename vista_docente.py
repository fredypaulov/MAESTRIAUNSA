# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                     VISTA DOCENTE - ANÁLISIS POR AULA                     ║
║        Análisis detallado, métricas ML, estudiantes desaprobados         ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

from constantes import INFO_INSTITUCION, ESCALA_CALIFICACIONES, UMBRAL_APROBACION
from procesamiento import obtener_columnas_notas, procesar_datos, procesar_datos_por_area
from analisis_ml import calcular_metricas_ml, calcular_matriz_confusion, interpretar_roc_auc, interpretar_f1_score
from visualizaciones import (
    crear_grafico_barras_horizontal,
    crear_grafico_barras_vertical,
    crear_histograma_distribucion,
    crear_grafico_matriz_confusion,
    generar_tabla_frecuencias
)
from utils import find_column, df_to_excel_bytes, calcular_porcentaje_seguro

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA: VISTA DOCENTE
# ═════════════════════════════════════════════════════════════════════════════

def pagina_vista_docente(datos_por_hoja: Dict[str, pd.DataFrame]):
    """
    👩‍🏫 VISTA DOCENTE COMPLETA
    
    Funcionalidades:
    - Análisis detallado por aula/salón
    - Clasificación por cursos/áreas curriculares
    - Estudiantes desaprobados con priorización
    - Métricas ML: ROC-AUC, F1-Score, Precision, Recall
    - Tabla de frecuencias por nivel
    - Análisis de tendencias
    - Seguimiento individualizado
    """
    
    st.title("👩‍🏫 Vista Docente: Análisis Detallado por Aula")
    st.caption(f"📍 {INFO_INSTITUCION.get('nombre_ie1', 'Institución Educativa')}")
    
    if not datos_por_hoja:
        st.warning("⚠️ No hay datos cargados. Por favor, cargue un archivo Excel desde la barra lateral.")
        
        st.info("""
        ### ✨ Funcionalidades Disponibles:
        
        - ✅ **Análisis detallado por estudiante en el aula**
        - ✅ **Seguimiento individualizado de progreso**
        - ✅ **Generación de informes de tutoría**
        - ✅ **Registro de observaciones pedagógicas**
        - ✅ **Clasificación por cursos y áreas**
        - ✅ **Identificación de estudiantes desaprobados**
        - ✅ **Métricas avanzadas (ROC-AUC, F1-Score)**
        - ✅ **Tabla de frecuencias por nivel**
        - ✅ **Análisis de tendencias y distribución**
        - ✅ **Comunicación con padres de familia**
        """)
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. SELECTOR DE AULA
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📚 Selección de Aula")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        nombre_aula = st.selectbox(
            "Seleccione el aula/grado:",
            list(datos_por_hoja.keys()),
            help="Seleccione el grado y sección a analizar"
        )
    
    df_aula = datos_por_hoja[nombre_aula]
    
    # Procesar datos
    with st.spinner(f"📊 Procesando datos de {nombre_aula}..."):
        columnas_notas, columnas_id = obtener_columnas_notas(df_aula)
        
        if not columnas_notas:
            st.error(f"❌ No se encontraron columnas de notas en '{nombre_aula}'")
            return
        
        df_procesado, columnas_num_proc = procesar_datos(df_aula, columnas_notas)
    
    with col2:
        total_est = len(df_procesado)
        promedio_aula = df_procesado['PROMEDIO'].mean()
        
        st.info(f"""
        **📊 Información del Aula:** {nombre_aula}  
        **👥 Total de estudiantes:** {total_est}  
        **📝 Áreas evaluadas:** {len(columnas_notas)}  
        **📈 Promedio general:** {promedio_aula:.2f}/20.00
        """)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. KPIs PRINCIPALES
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📊 Indicadores Clave del Aula")
    
    aprobados = (df_procesado['ESTADO'] == 'Aprobado').sum()
    desaprobados = total_est - aprobados
    tasa_aprobacion = calcular_porcentaje_seguro(aprobados, total_est)
    
    metricas_ml = calcular_metricas_ml(df_procesado, UMBRAL_APROBACION)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Estudiantes", f"{total_est} 🎓")
    col2.metric("Promedio Aula", f"{promedio_aula:.2f}", delta="sobre 20")
    col3.metric("Aprobados", f"{aprobados} ✅", delta=f"{tasa_aprobacion:.1f}%")
    col4.metric("Desaprobados", f"{desaprobados} ⚠️", 
                delta=f"{calcular_porcentaje_seguro(desaprobados, total_est):.1f}%", 
                delta_color="inverse")
    
    if metricas_ml:
        col5.metric("F1-Score", f"{metricas_ml.get('f1_score', 0):.3f}", 
                   help="Métrica de precisión balanceada")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. TABLA DE FRECUENCIAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📈 Tabla de Frecuencias por Nivel de Logro")
    
    df_frecuencias = generar_tabla_frecuencias(df_procesado)
    
    col_freq1, col_freq2 = st.columns([1, 2])
    
    with col_freq1:
        st.dataframe(
            df_frecuencias,
            use_container_width=True,
            hide_index=True,
            column_config={
                'NIVEL': st.column_config.TextColumn('Nivel', width='small'),
                'DESCRIPCIÓN': st.column_config.TextColumn('Descripción', width='medium'),
                'ESTUDIANTES': st.column_config.NumberColumn('Cantidad', format='%d'),
                'PORCENTAJE': st.column_config.NumberColumn('Porcentaje', format='%.2f%%')
            }
        )
    
    with col_freq2:
        from constantes import COLORES_NIVELES
        fig_barras = crear_grafico_barras_horizontal(
            df_frecuencias, 
            'ESTUDIANTES', 
            'NIVEL', 
            f'Distribución de Niveles en {nombre_aula}',
            'NIVEL'
        )
        st.plotly_chart(fig_barras, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. MÉTRICAS ML
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🤖 Métricas Avanzadas de Evaluación (Machine Learning)")
    
    if metricas_ml:
        col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
        
        with col_ml1:
            roc_auc = metricas_ml.get('roc_auc', 0)
            nivel, color, mensaje = interpretar_roc_auc(roc_auc)
            
            st.metric("ROC-AUC Score", f"{roc_auc:.3f}", 
                     help="Área bajo la curva ROC. Valor óptimo: 1.0")
            
            if color == "success":
                st.success(mensaje)
            elif color == "info":
                st.info(mensaje)
            else:
                st.warning(mensaje)
        
        with col_ml2:
            f1 = metricas_ml.get('f1_score', 0)
            st.metric("F1-Score", f"{f1:.3f}", 
                     help="Balance entre Precision y Recall")
        
        with col_ml3:
            precision = metricas_ml.get('precision', 0)
            st.metric("Precision", f"{precision:.3f}", 
                     help="Proporción de predicciones positivas correctas")
        
        with col_ml4:
            recall = metricas_ml.get('recall', 0)
            st.metric("Recall", f"{recall:.3f}", 
                     help="Proporción de casos positivos detectados")
        
        with st.expander("📖 ¿Qué significan estas métricas?"):
            st.markdown("""
            ### Interpretación de Métricas ML
            
            **ROC-AUC (Receiver Operating Characteristic):**
            - Mide la capacidad de distinguir entre aprobados y desaprobados
            - **1.0 = Perfecto:** Clasificación perfecta
            - **0.9-1.0 = Excelente:** Alta precisión
            - **0.7-0.9 = Bueno:** Precisión aceptable
            - **< 0.7 = Regular:** Requiere mejoras
            
            **F1-Score:**
            - Métrica balanceada (combina precision y recall)
            - Útil cuando hay desbalance entre clases
            - Valor ideal: cercano a 1.0
            
            **Precision:**
            - De los predichos como "aprobados", ¿cuántos lo están realmente?
            - Alta precision = Pocas falsas alarmas
            
            **Recall (Sensibilidad):**
            - De todos los aprobados reales, ¿cuántos fueron detectados?
            - Alto recall = No se escapan casos positivos
            
            📚 **Ref:** MINEDU - Evaluación formativa y predictiva
            """)
    else:
        st.warning("⚠️ No se pudieron calcular métricas ML. Instale scikit-learn: `pip install scikit-learn`")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. ESTUDIANTES DESAPROBADOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🚨 Estudiantes Desaprobados - Requieren Reforzamiento Académico")
    
    df_desaprobados = df_procesado[df_procesado['ESTADO'] == 'Desaprobado'].copy()
    
    if len(df_desaprobados) > 0:
        col_nombre = find_column(df_desaprobados, ['APELLIDOS', 'NOMBRES', 'ESTUDIANTE'])
        
        # Clasificar por prioridad
        df_desaprobados['PRIORIDAD'] = df_desaprobados['CALIFICACION_LETRA'].map({
            'C': '🔴 CRÍTICO',
            'B': '🟡 MODERADO'
        })
        
        df_desaprobados = df_desaprobados.sort_values('PROMEDIO')
        
        col_des1, col_des2 = st.columns([1, 3])
        
        with col_des1:
            pct_desap = calcular_porcentaje_seguro(len(df_desaprobados), total_est)
            st.metric("Total Desaprobados", len(df_desaprobados), 
                     delta=f"{pct_desap:.1f}%", delta_color="inverse")
            
            criticos = (df_desaprobados['CALIFICACION_LETRA'] == 'C').sum()
            moderados = (df_desaprobados['CALIFICACION_LETRA'] == 'B').sum()
            
            st.markdown(f"""
            **Clasificación:**
            - 🔴 **Críticos (C):** {criticos}
            - 🟡 **Moderados (B):** {moderados}
            """)
        
        with col_des2:
            cols_mostrar = []
            if col_nombre:
                cols_mostrar.append(col_nombre)
            cols_mostrar.extend(['PROMEDIO', 'CALIFICACION_LETRA', 'PRIORIDAD'])
            
            st.dataframe(
                df_desaprobados[cols_mostrar].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    col_nombre: st.column_config.TextColumn('Estudiante', width='large') if col_nombre else None,
                    'PROMEDIO': st.column_config.NumberColumn('Promedio', format='%.2f'),
                    'CALIFICACION_LETRA': st.column_config.TextColumn('Nivel', width='small'),
                    'PRIORIDAD': st.column_config.TextColumn('Prioridad', width='medium')
                }
            )
        
        # Plan de acción
        with st.expander("📋 Plan de Acción para Estudiantes Desaprobados"):
            st.markdown("""
            ### Plan de Reforzamiento Académico MINEDU
            
            #### 🔴 Para Nivel C (Crítico):
            1. **Evaluación diagnóstica inmediata**
            2. **Tutorías personalizadas:** 3 sesiones/semana (45 min)
            3. **Material adaptado:** Fichas con ejemplos concretos
            4. **Reunión con padres:** Compromiso familiar
            5. **Seguimiento semanal:** Registro de avances
            
            #### 🟡 Para Nivel B (Moderado):
            1. **Reforzamiento grupal:** 2 sesiones/semana
            2. **Aprendizaje entre pares:** Tutorías estudiante-estudiante
            3. **Tareas diferenciadas:** Ejercicios graduados
            4. **Retroalimentación constante:** Comentarios específicos
            
            📚 **Ref:** MINEDU - Orientaciones para reforzamiento escolar 2024-2025
            """)
        
        # Descarga
        csv_desap = df_desaprobados[cols_mostrar].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 Descargar Lista de Desaprobados (CSV)",
            data=csv_desap,
            file_name=f"desaprobados_{nombre_aula}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ ¡Excelente! No hay estudiantes desaprobados en esta aula.")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6. ANÁLISIS POR ÁREA/CURSO
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📚 Análisis por Curso/Área Curricular")
    
    areas_map = procesar_datos_por_area(df_procesado, columnas_num_proc)
    
    if areas_map:
        promedios_por_area = {}
        for area, cols in areas_map.items():
            if cols:
                try:
                    promedio_area = df_procesado[cols].mean(axis=1).mean()
                    promedios_por_area[area] = promedio_area
                except:
                    continue
        
        if promedios_por_area:
            df_areas = pd.DataFrame(
                list(promedios_por_area.items()), 
                columns=['Área', 'Promedio']
            ).sort_values('Promedio', ascending=False)
            
            col_area1, col_area2 = st.columns([1, 2])
            
            with col_area1:
                st.dataframe(
                    df_areas,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Área': st.column_config.TextColumn('Área Curricular'),
                        'Promedio': st.column_config.NumberColumn('Promedio', format='%.2f')
                    }
                )
            
            with col_area2:
                fig_areas = crear_grafico_barras_vertical(
                    df_areas, 
                    'Promedio', 
                    'Área', 
                    'Rendimiento Promedio por Área'
                )
                st.plotly_chart(fig_areas, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 7. ANÁLISIS DE TENDENCIAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📈 Análisis de Tendencias y Distribución")
    
    tab1, tab2, tab3 = st.tabs(["📊 Histograma", "📈 Distribución", "🎯 Matriz Confusión"])
    
    with tab1:
        fig_hist = crear_histograma_distribucion(df_procesado, promedio_aula, UMBRAL_APROBACION)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        import plotly.express as px
        fig_box = px.box(
            df_procesado,
            y='PROMEDIO',
            title='Diagrama de Caja - Distribución de Promedios'
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        cm = calcular_matriz_confusion(df_procesado, UMBRAL_APROBACION)
        fig_cm = crear_grafico_matriz_confusion(cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.info("""
        **Interpretación:**
        - **Diagonal principal:** Clasificaciones correctas
        - **Fuera de diagonal:** Errores de clasificación
        """)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 8. LISTA COMPLETA CON SEGUIMIENTO
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📝 Lista Completa de Estudiantes - Seguimiento Individualizado")
    
    col_nombre = find_column(df_procesado, ['APELLIDOS', 'NOMBRES', 'ESTUDIANTE'])
    
    if col_nombre:
        cols_vista = [col_nombre, 'PROMEDIO', 'CALIFICACION_LETRA', 'ESTADO']
        df_vista = df_procesado[cols_vista].copy()
        df_vista = df_vista.sort_values('PROMEDIO', ascending=False).reset_index(drop=True)
        df_vista.index += 1
        
        df_vista['OBSERVACIÓN'] = df_vista['CALIFICACION_LETRA'].map({
            'AD': '⭐ Logro Destacado',
            'A': '✅ Logro Esperado',
            'B': '⚠️ En Proceso - Reforzar',
            'C': '🚨 En Inicio - Reforzamiento Urgente'
        })
        
        st.dataframe(df_vista, use_container_width=True, height=400)
        
        col_desc1, col_desc2 = st.columns(2)
        
        with col_desc1:
            csv_completo = df_vista.to_csv(index=True, index_label='N°', encoding='utf-8-sig')
            st.download_button(
                label="📥 Descargar Lista Completa (CSV)",
                data=csv_completo,
                file_name=f"lista_{nombre_aula}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_desc2:
            excel_completo = df_to_excel_bytes(df_vista, f"Lista_{nombre_aula}")
            st.download_button(
                label="📊 Descargar Lista Excel",
                data=excel_completo,
                file_name=f"lista_{nombre_aula}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 9. RECOMENDACIONES PEDAGÓGICAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 💡 Recomendaciones Pedagógicas para el Docente")
    
    pct_desap = calcular_porcentaje_seguro(desaprobados, total_est)
    pct_c = calcular_porcentaje_seguro((df_procesado['CALIFICACION_LETRA'] == 'C').sum(), total_est)
    
    if pct_desap > 30:
        st.error(f"""
        🚨 **Situación Crítica: {pct_desap:.1f}% de desaprobación**
        
        **Acciones Inmediatas:**
        1. Reunión urgente con dirección
        2. Revisión del plan curricular
        3. Programa intensivo de reforzamiento (5+ horas/semana)
        4. Solicitar apoyo especializado
        5. Reunión con todos los padres de familia
        
        📚 **Ref:** MINEDU - Protocolo de intervención pedagógica
        """)
    elif pct_desap > 15:
        st.warning(f"""
        ⚠️ **Requiere Atención: {pct_desap:.1f}% de desaprobación**
        
        **Acciones Recomendadas:**
        1. Reforzar evaluación formativa
        2. Tutorías grupales 2-3 veces/semana
        3. Comunicación constante con padres
        4. Adaptación de materiales didácticos
        
        📚 **Ref:** MINEDU - Acompañamiento pedagógico
        """)
    else:
        st.success(f"""
        ✅ **Buen Rendimiento: Solo {pct_desap:.1f}% de desaprobación**
        
        **Continuar con:**
        1. Estrategias actuales (están funcionando)
        2. Monitoreo preventivo
        3. Desafíos para estudiantes destacados
        4. Aprendizaje colaborativo
        
        📚 **Ref:** MINEDU - Buenas prácticas pedagógicas
        """)
    
    st.success("✅ Vista Docente cargada correctamente con todas las funcionalidades activas.")
