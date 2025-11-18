
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║          VISTA ANÁLISIS DE ESTUDIANTES PRIORIZADOS - VERSIÓN CORREGIDA   ║
║        🎯 Sistema con Manejo Robusto de Errores                          ║
║        ✅ Compatible con cualquier formato de columnas                   ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import io

from constantes import (
    INFO_INSTITUCION,
    ESCALA_CALIFICACIONES,
    ESTRATEGIAS_MINEDU,
    UMBRAL_APROBACION,
    AREAS_CURRICULARES,
    EQUIVALENCIAS_NOTAS,
    COLORES_NIVELES
)
from procesamiento import obtener_columnas_notas, procesar_datos, procesar_datos_por_area
from contexto import gestor_evaluacion
from utils import find_column, df_to_excel_bytes, calcular_porcentaje_seguro
from analisis_ml import calcular_metricas_ml, calcular_matriz_confusion
from visualizaciones import (
    crear_grafico_pastel_niveles,
    crear_mapa_calor_areas,
    generar_tabla_frecuencias
)

# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ANÁLISIS AVANZADO - VERSIÓN CORREGIDA
# ═════════════════════════════════════════════════════════════════════════════

def identificar_estudiantes_priorizados(df: pd.DataFrame, umbral: float = 11.0) -> pd.DataFrame:
    """
    Identifica estudiantes que requieren reforzamiento académico urgente
    VERSIÓN CORREGIDA: Manejo robusto de tipos de columnas
    
    Criterios:
    - Promedio < 11 (Desaprobado)
    - Nivel C (En Inicio)
    - Más de 3 áreas desaprobadas
    
    Args:
        df: DataFrame con datos procesados
        umbral: Promedio mínimo de aprobación
        
    Returns:
        DataFrame con estudiantes priorizados
    """
    try:
        # Estudiantes desaprobados
        mask_desaprobados = df['PROMEDIO'] < umbral
        
        # Estudiantes en nivel C
        mask_nivel_c = df['CALIFICACION_LETRA'] == 'C'
        
        # Unir condiciones
        mask_priorizados = mask_desaprobados | mask_nivel_c
        
        df_priorizados = df[mask_priorizados].copy()
        
        # Calcular cantidad de áreas desaprobadas
        # CORRECCIÓN: Convertir nombres de columnas a string antes de verificar
        columnas_numericas = [
            col for col in df.columns 
            if isinstance(col, str) and col.endswith('_num')
        ]
        
        # Si no hay columnas con '_num', buscar columnas numéricas alternativas
        if not columnas_numericas:
            columnas_numericas = [
                col for col in df.columns 
                if isinstance(col, (int, float)) or 
                (isinstance(col, str) and any(char.isdigit() for char in str(col)))
            ]
        
        if columnas_numericas and len(columnas_numericas) > 0:
            try:
                df_priorizados['AREAS_DESAPROBADAS'] = (df_priorizados[columnas_numericas] < umbral).sum(axis=1)
            except Exception as e:
                st.warning(f"No se pudo calcular áreas desaprobadas: {e}")
                df_priorizados['AREAS_DESAPROBADAS'] = 0
        else:
            df_priorizados['AREAS_DESAPROBADAS'] = 0
        
        # Clasificar prioridad
        def clasificar_prioridad(row):
            if row['PROMEDIO'] < 8:
                return '🔴 CRÍTICO'
            elif row['PROMEDIO'] < 11:
                return '🟠 ALTO'
            elif row['CALIFICACION_LETRA'] == 'C':
                return '🟡 MEDIO'
            else:
                return '🟢 BAJO'
        
        df_priorizados['PRIORIDAD'] = df_priorizados.apply(clasificar_prioridad, axis=1)
        
        return df_priorizados.sort_values(['PRIORIDAD', 'PROMEDIO'])
    
    except Exception as e:
        st.error(f"Error al identificar estudiantes priorizados: {e}")
        # Retornar DataFrame vacío en caso de error
        return pd.DataFrame(columns=['PROMEDIO', 'CALIFICACION_LETRA', 'AREAS_DESAPROBADAS', 'PRIORIDAD'])

def analizar_por_area(df: pd.DataFrame, columnas_num: List[str], areas_map: Dict) -> pd.DataFrame:
    """
    Analiza el rendimiento por área curricular
    VERSIÓN CORREGIDA: Manejo robusto de errores
    
    Returns:
        DataFrame con análisis por área
    """
    resultados = []
    
    for area, cols in areas_map.items():
        if not cols:
            continue
        
        try:
            # Filtrar solo columnas que existen en el DataFrame
            cols_existentes = [col for col in cols if col in df.columns]
            
            if not cols_existentes:
                continue
            
            # Calcular estadísticas por área
            notas_area = df[cols_existentes].values.flatten()
            notas_area = notas_area[~np.isnan(notas_area)]
            
            if len(notas_area) == 0:
                continue
            
            promedio_area = notas_area.mean()
            desaprobados_area = (notas_area < UMBRAL_APROBACION).sum()
            total_evaluaciones = len(notas_area)
            tasa_desaprobacion = calcular_porcentaje_seguro(desaprobados_area, total_evaluaciones)
            
            resultados.append({
                'Área': area,
                'Promedio': round(promedio_area, 2),
                'Total_Evaluaciones': total_evaluaciones,
                'Desaprobados': desaprobados_area,
                'Tasa_Desaprobación_%': tasa_desaprobacion,
                'Nivel_Riesgo': '🔴 Alto' if tasa_desaprobacion > 30 else '🟡 Medio' if tasa_desaprobacion > 15 else '🟢 Bajo'
            })
        except Exception as e:
            st.warning(f"Error analizando área {area}: {e}")
            continue
    
    if resultados:
        df_areas = pd.DataFrame(resultados).sort_values('Tasa_Desaprobación_%', ascending=False)
        return df_areas
    
    return pd.DataFrame()

def generar_recomendaciones_integral(
    df_priorizados: pd.DataFrame,
    metricas_ml: Dict,
    areas_criticas: List[str]
) -> Dict[str, str]:
    """
    Genera recomendaciones personalizadas para todos los actores educativos
    VERSIÓN CORREGIDA: Manejo seguro de datos
    """
    try:
        total_priorizados = len(df_priorizados)
        criticos = (df_priorizados['PRIORIDAD'] == '🔴 CRÍTICO').sum() if 'PRIORIDAD' in df_priorizados.columns else 0
    except Exception as e:
        st.warning(f"Error calculando totales: {e}")
        total_priorizados = 0
        criticos = 0
    
    recomendaciones = {}
    
    # ═══════════════════════════════════════════════════════════════════════
    # RECOMENDACIONES PARA ESTUDIANTES
    # ═══════════════════════════════════════════════════════════════════════
    
    recomendaciones['estudiantes'] = f"""
### 🎓 Recomendaciones para los Estudiantes

**Situación Actual:** {total_priorizados} estudiantes requieren reforzamiento académico.

#### ✅ Acciones Inmediatas que Debes Tomar:

1. **Organiza tu Tiempo de Estudio:**
   - Crea un horario de estudio diario (mínimo 2 horas)
   - Estudia en un lugar tranquilo y bien iluminado
   - Evita distracciones (celular, TV, videojuegos)
   - Usa la técnica Pomodoro: 25 minutos de estudio + 5 de descanso

2. **Técnicas de Estudio Efectivas:**
   - **Resúmenes y mapas mentales:** Organiza la información visualmente
   - **Fichas de estudio:** Crea tarjetas con preguntas y respuestas
   - **Explica lo aprendido:** Enseña a un familiar lo que estudiaste
   - **Práctica constante:** Resuelve ejercicios todos los días

3. **Busca Ayuda Cuando la Necesites:**
   - Pregunta a tus profesores en clase
   - Asiste a las tutorías y reforzamiento
   - Forma grupos de estudio con compañeros
   - Usa recursos educativos en línea

4. **Cuida tu Salud:**
   - Duerme 8 horas diarias
   - Aliméntate bien
   - Haz ejercicio
   - Habla si te sientes estresado

5. **Áreas Prioritarias:**
   {', '.join(areas_criticas[:3]) if areas_criticas else 'Todas las áreas principales'}

#### 💪 Recuerda:
¡Tú puedes mejorar! El rendimiento no define tu valor. Cada día es una oportunidad.

#### 🌐 Recursos Gratuitos:
- **PerúEduca:** https://www.perueduca.pe
- **Khan Academy:** https://es.khanacademy.org
- **Aprendo en Casa:** https://aprendoencasa.pe
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RECOMENDACIONES PARA DOCENTES
    # ═══════════════════════════════════════════════════════════════════════
    
    nivel_urgencia = "CRÍTICA" if criticos > 5 else "ALTA" if total_priorizados > 10 else "MODERADA"
    
    recomendaciones['docentes'] = f"""
### 👩‍🏫 Recomendaciones para los Docentes

**Situación:** {total_priorizados} estudiantes priorizados ({criticos} críticos)
**Urgencia:** {nivel_urgencia}

#### 📋 Plan de Intervención:

1. **Evaluación Diagnóstica Urgente:**
   - Aplicar prueba de entrada
   - Identificar brechas específicas
   - Evaluar prerrequisitos

2. **Estrategias Diferenciadas:**
   
   **Críticos (🔴):**
   - Tutorías 3 veces/semana
   - Material gradual
   - Evaluación formativa diaria
   
   **Alto Riesgo (🟠):**
   - Reforzamiento grupal 2-3/semana
   - Fichas diferenciadas
   - Retroalimentación semanal

3. **Adaptaciones Curriculares:**
   - Material en diferentes formatos
   - Priorizar competencias esenciales
   - Diversificar evaluaciones

4. **Áreas Críticas:**
   {', '.join(areas_criticas[:3]) if areas_criticas else 'Por definir'}

#### 📚 Referencia MINEDU:
- RVM N° 094-2020-MINEDU
- Guía de Reforzamiento 2025
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RECOMENDACIONES PARA PADRES
    # ═══════════════════════════════════════════════════════════════════════
    
    recomendaciones['padres'] = f"""
### 👨‍👩‍👧‍👦 Recomendaciones para Padres

**Estimados Padres:**

Su hijo(a) requiere reforzamiento académico. Su apoyo es fundamental.

#### ❤️ Apoyo Emocional:

1. **Actitud Positiva:**
   - ❌ Evitar: Castigar, comparar, etiquetar
   - ✅ Hacer: Confiar, reconocer esfuerzos
   - "Confío en ti, puedes mejorar"

2. **Comunicación:**
   - Preguntar cómo se siente
   - Escuchar sin juzgar
   - Identificar problemas

#### 📚 Apoyo Académico:

1. **Ambiente de Estudio:**
   - Espacio tranquilo
   - Sin distractores
   - Horario fijo

2. **Supervisión:**
   - NO hacer tareas por ellos
   - SÍ guiar con preguntas
   - Revisar cuadernos diariamente

3. **Rutina Sugerida:**
   - 3:30-4:30 PM: Tareas
   - 4:45-5:45 PM: Repaso
   - Fin de semana: Reforzamiento

#### 🏥 Salud:

1. **Física:**
   - Desayuno nutritivo
   - 8 horas de sueño
   - Ejercicio regular

2. **Mental:**
   - Tiempo en familia
   - Observar señales de estrés
   - Buscar apoyo si es necesario

#### 📞 Recursos:
- Línea 113 - Salud Mental
- SíseVe: www.siseve.pe

---

**💪 ¡Juntos lo lograremos!**
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RECOMENDACIONES PARA DIRECTOR
    # ═══════════════════════════════════════════════════════════════════════
    
    f1_score = metricas_ml.get('f1_score', 0) if metricas_ml else 0
    roc_auc = metricas_ml.get('roc_auc', 0) if metricas_ml else 0
    
    recomendaciones['director'] = f"""
### 🏛️ Recomendaciones para la Dirección

**Informe Ejecutivo**

#### 📊 Indicadores:

- **Priorizados:** {total_priorizados}
- **Críticos:** {criticos}
- **Áreas Críticas:** {len(areas_criticas)}
- **Métricas ML:**
  - F1-Score: {f1_score:.3f}
  - ROC-AUC: {roc_auc:.3f}

#### 🎯 Plan de Acción:

1. **Esta Semana:**
   - Reunión coordinación académica
   - Citar padres de estudiantes críticos
   - Reorganizar recursos

2. **Próximas 4 Semanas:**
   - Reforzamiento estructurado
   - Monitoreo docente
   - Sistema de alerta

3. **Coordinaciones:**
   - UGEL Arequipa Sur
   - Sector Salud
   - Comunidad

#### 📈 Indicadores de Éxito:
- 60% suben nivel
- 50% reducción críticos
- 20% aumento promedio

#### 📋 Marco Normativo:
- Ley N° 28044
- RVM N° 094-2020-MINEDU
- DS N° 004-2018-MINEDU

---

**Situación manejable con acciones coordinadas.**
    """
    
    return recomendaciones

def generar_excel_completo_priorizados(
    df_priorizados: pd.DataFrame,
    df_analisis_areas: pd.DataFrame,
    metricas_ml: Dict,
    recomendaciones: Dict,
    col_nombre: str
) -> bytes:
    """
    Genera archivo Excel profesional
    VERSIÓN CORREGIDA: Manejo robusto de errores
    """
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Formatos
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            critico_format = workbook.add_format({
                'bg_color': '#FF6B6B',
                'font_color': 'white',
                'bold': True
            })
            
            # Hoja 1: Estudiantes Priorizados
            cols_export = [col for col in [col_nombre, 'AULA', 'PROMEDIO', 'CALIFICACION_LETRA', 
                          'ESTADO', 'AREAS_DESAPROBADAS', 'PRIORIDAD'] if col in df_priorizados.columns]
            
            if cols_export:
                df_export = df_priorizados[cols_export].copy()
                df_export.to_excel(writer, sheet_name='Estudiantes_Priorizados', index=False)
                
                worksheet1 = writer.sheets['Estudiantes_Priorizados']
                
                # Formato encabezados
                for col_num, value in enumerate(df_export.columns.values):
                    worksheet1.write(0, col_num, value, header_format)
                
                # Ajustar anchos
                worksheet1.set_column('A:A', 40)
                worksheet1.set_column('B:G', 15)
            
            # Hoja 2: Análisis por Área
            if not df_analisis_areas.empty:
                df_analisis_areas.to_excel(writer, sheet_name='Analisis_por_Area', index=False)
                worksheet2 = writer.sheets['Analisis_por_Area']
                
                for col_num, value in enumerate(df_analisis_areas.columns.values):
                    worksheet2.write(0, col_num, value, header_format)
                
                worksheet2.set_column('A:A', 30)
                worksheet2.set_column('B:F', 18)
            
            # Hoja 3: Métricas ML
            if metricas_ml:
                df_metricas = pd.DataFrame([{
                    'Métrica': 'F1-Score',
                    'Valor': metricas_ml.get('f1_score', 0),
                    'Interpretación': 'Balance precision-recall'
                }, {
                    'Métrica': 'ROC-AUC',
                    'Valor': metricas_ml.get('roc_auc', 0),
                    'Interpretación': 'Capacidad discriminación'
                }])
                
                df_metricas.to_excel(writer, sheet_name='Metricas_Calidad', index=False)
            
            # Hojas 4-7: Recomendaciones
            for sheet_name, key in [
                ('Rec_Estudiantes', 'estudiantes'),
                ('Rec_Docentes', 'docentes'),
                ('Rec_Padres', 'padres'),
                ('Rec_Director', 'director')
            ]:
                df_rec = pd.DataFrame([{
                    'Tipo': f'Recomendaciones para {key}',
                    'Contenido': recomendaciones.get(key, '')
                }])
                df_rec.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Hoja 8: Info
            df_info = pd.DataFrame([{
                'Campo': 'Institución',
                'Valor': INFO_INSTITUCION.get('nombre_ie1', '')
            }, {
                'Campo': 'Fecha',
                'Valor': datetime.now().strftime('%d/%m/%Y %H:%M')
            }, {
                'Campo': 'Total Priorizados',
                'Valor': len(df_priorizados)
            }])
            
            df_info.to_excel(writer, sheet_name='Info_Reporte', index=False)
        
        return output.getvalue()
    
    except Exception as e:
        st.error(f"Error generando Excel: {e}")
        return b""

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL - VERSIÓN CORREGIDA
# ═════════════════════════════════════════════════════════════════════════════

def pagina_analisis_priorizados(datos_raw: Optional[Dict[str, pd.DataFrame]]):
    """
    🎯 ANÁLISIS DE ESTUDIANTES PRIORIZADOS
    VERSIÓN CORREGIDA CON MANEJO ROBUSTO DE ERRORES
    """
    
    st.title("🎯 Análisis de Estudiantes Priorizados")
    st.caption(f"📍 {INFO_INSTITUCION.get('nombre_ie1', 'Institución Educativa')}")
    
    if not datos_raw:
        st.warning("⚠️ No hay datos cargados.")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. PROCESAMIENTO
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📊 Procesamiento de Datos")
    
    opcion = st.radio(
        "Alcance:",
        ["📊 Global", "🎯 Por Aula"],
        horizontal=True
    )
    
    if opcion == "🎯 Por Aula":
        hoja = st.selectbox("Aula:", list(datos_raw.keys()))
        hojas = {hoja: datos_raw[hoja]}
    else:
        hojas = datos_raw
    
    df_list = []
    
    with st.spinner("🔄 Procesando..."):
        for nombre, df_hoja in hojas.items():
            try:
                cols_notas, _ = obtener_columnas_notas(df_hoja)
                if not cols_notas:
                    continue
                
                df_proc, cols_num = procesar_datos(df_hoja, cols_notas)
                df_proc['AULA'] = nombre
                df_list.append((df_proc, cols_num))
            except Exception as e:
                st.warning(f"⚠️ Error en '{nombre}': {e}")
                continue
    
    if not df_list:
        st.error("❌ No se pudieron procesar datos")
        return
    
    df_consolidado = pd.concat([df for df, _ in df_list], ignore_index=True)
    columnas_num = df_list[0][1]
    
    st.success(f"✅ {len(df_consolidado)} estudiantes procesados")
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. IDENTIFICACIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🚨 Estudiantes en Riesgo")
    
    df_priorizados = identificar_estudiantes_priorizados(df_consolidado, UMBRAL_APROBACION)
    
    if df_priorizados.empty:
        st.success("🎉 ¡No hay estudiantes en riesgo!")
        return
    
    total_est = len(df_consolidado)
    total_prior = len(df_priorizados)
    pct = calcular_porcentaje_seguro(total_prior, total_est)
    
    criticos = (df_priorizados['PRIORIDAD'] == '🔴 CRÍTICO').sum() if 'PRIORIDAD' in df_priorizados.columns else 0
    altos = (df_priorizados['PRIORIDAD'] == '🟠 ALTO').sum() if 'PRIORIDAD' in df_priorizados.columns else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Priorizados", total_prior, f"{pct:.1f}%")
    col2.metric("🔴 Críticos", criticos)
    col3.metric("🟠 Alto Riesgo", altos)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. ANÁLISIS POR ÁREA
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📚 Análisis por Área")
    
    areas_map = procesar_datos_por_area(df_consolidado, columnas_num)
    df_areas = analizar_por_area(df_consolidado, columnas_num, areas_map)
    
    if not df_areas.empty:
        col_a1, col_a2 = st.columns([1, 2])
        
        with col_a1:
            st.dataframe(df_areas, use_container_width=True, hide_index=True)
        
        with col_a2:
            fig = px.bar(
                df_areas.head(8),
                x='Tasa_Desaprobación_%',
                y='Área',
                orientation='h',
                title='Tasa Desaprobación (%)',
                color='Tasa_Desaprobación_%',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    areas_criticas = df_areas.head(3)['Área'].tolist() if not df_areas.empty else []
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. VISUALIZACIONES
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📊 Visualizaciones")
    
    tab1, tab2, tab3 = st.tabs(["📈 Frecuencias", "🥧 Pastel", "🤖 Métricas ML"])
    
    with tab1:
        df_freq = generar_tabla_frecuencias(df_priorizados)
        st.dataframe(df_freq, use_container_width=True, hide_index=True)
    
    with tab2:
        fig_pastel = crear_grafico_pastel_niveles(df_freq, 'Distribución Priorizados')
        st.plotly_chart(fig_pastel, use_container_width=True)
    
    with tab3:
        metricas = calcular_metricas_ml(df_consolidado, UMBRAL_APROBACION)
        
        if metricas:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("F1-Score", f"{metricas.get('f1_score', 0):.3f}")
            col_m2.metric("ROC-AUC", f"{metricas.get('roc_auc', 0):.3f}")
            col_m3.metric("Precision", f"{metricas.get('precision', 0):.3f}")
        else:
            st.info("Métricas ML no disponibles")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. LISTA DETALLADA
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📋 Lista Detallada")
    
    col_nombre = find_column(df_priorizados, ['APELLIDOS', 'NOMBRES', 'ESTUDIANTE'])
    
    if col_nombre:
        cols = [c for c in [col_nombre, 'AULA', 'PROMEDIO', 'CALIFICACION_LETRA', 
                'ESTADO', 'PRIORIDAD'] if c in df_priorizados.columns]
        
        df_mostrar = df_priorizados[cols].reset_index(drop=True)
        df_mostrar.index += 1
        
        st.dataframe(df_mostrar, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6. RECOMENDACIONES
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 💡 Recomendaciones MINEDU 2025")
    
    recs = generar_recomendaciones_integral(
        df_priorizados,
        metricas if metricas else {},
        areas_criticas
    )
    
    tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs([
        "🎓 Estudiantes",
        "👩‍🏫 Docentes",
        "👨‍👩‍👧‍👦 Padres",
        "🏛️ Director"
    ])
    
    with tab_r1:
        st.markdown(recs['estudiantes'])
    with tab_r2:
        st.markdown(recs['docentes'])
    with tab_r3:
        st.markdown(recs['padres'])
    with tab_r4:
        st.markdown(recs['director'])
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 7. EXPORTACIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📥 Exportar")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        if col_nombre:
            excel = generar_excel_completo_priorizados(
                df_priorizados,
                df_areas,
                metricas if metricas else {},
                recs,
                col_nombre
            )
            
            if excel:
                st.download_button(
                    "📊 Descargar Excel",
                    data=excel,
                    file_name=f"Priorizados_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    with col_e2:
        if col_nombre:
            csv = df_mostrar.to_csv(index=True, encoding='utf-8-sig')
            st.download_button(
                "📄 Descargar CSV",
                data=csv,
                file_name=f"Priorizados_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.success(f"✅ Análisis completado: {total_prior} estudiantes priorizados identificados")
