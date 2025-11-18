# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║              VISTA ANÁLISIS ESTUDIANTIL - MINEDU 2025                     ║
║        🧑‍🎓 Análisis Individual por Estudiante con IA Predictiva         ║
║        📊 Estrategias de Reforzamiento Personalizadas                    ║
║        ✅ Sistema de Alerta Temprana Académica                           ║
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

from constantes import (
    INFO_INSTITUCION, 
    ESCALA_CALIFICACIONES, 
    ESTRATEGIAS_MINEDU,
    UMBRAL_APROBACION,
    AREAS_CURRICULARES
)
from procesamiento import obtener_columnas_notas, procesar_datos, procesar_datos_por_area
from contexto import gestor_evaluacion
from utils import find_column, df_to_excel_bytes, calcular_porcentaje_seguro
from visualizaciones import generar_tabla_frecuencias

# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES ESPECÍFICAS
# ═════════════════════════════════════════════════════════════════════════════

def buscar_estudiante(df: pd.DataFrame, termino_busqueda: str, col_nombre: str) -> pd.DataFrame:
    """
    Busca estudiante por nombre o código
    
    Args:
        df: DataFrame con datos de estudiantes
        termino_busqueda: Nombre o código a buscar
        col_nombre: Nombre de la columna de identificación
        
    Returns:
        DataFrame filtrado con resultados
    """
    if not termino_busqueda or not col_nombre:
        return pd.DataFrame()
    
    termino = termino_busqueda.upper().strip()
    
    # Buscar en la columna de nombres
    mascara = df[col_nombre].astype(str).str.upper().str.contains(termino, na=False)
    
    # También buscar en índice si parece ser un código numérico
    if termino.isdigit():
        mascara = mascara | (df.index == int(termino))
    
    return df[mascara]

def calcular_tendencia(notas: List[float]) -> str:
    """
    Calcula la tendencia de las notas (mejorando, estable, decayendo)
    
    Args:
        notas: Lista de notas ordenadas cronológicamente
        
    Returns:
        Tendencia detectada
    """
    if len(notas) < 2:
        return "📊 Sin datos suficientes"
    
    # Calcular diferencias
    diferencias = [notas[i+1] - notas[i] for i in range(len(notas)-1)]
    promedio_dif = sum(diferencias) / len(diferencias)
    
    if promedio_dif > 0.5:
        return "📈 Mejorando (tendencia positiva)"
    elif promedio_dif < -0.5:
        return "📉 Decayendo (tendencia negativa) ⚠️"
    else:
        return "➡️ Estable (sin cambios significativos)"

def predecir_riesgo_academico(promedio_actual: float, tendencia: str, nivel_actual: str) -> Dict:
    """
    Predice el riesgo de desaprobación futura del estudiante
    
    Args:
        promedio_actual: Promedio actual del estudiante
        tendencia: Tendencia de notas
        nivel_actual: Nivel actual (AD/A/B/C)
        
    Returns:
        Diccionario con nivel de riesgo y recomendaciones
    """
    riesgo_score = 0
    
    # Factor 1: Promedio actual
    if promedio_actual < 11:
        riesgo_score += 40
    elif promedio_actual < 13:
        riesgo_score += 25
    elif promedio_actual < 15:
        riesgo_score += 10
    
    # Factor 2: Tendencia
    if "Decayendo" in tendencia:
        riesgo_score += 30
    elif "Estable" in tendencia and promedio_actual < 13:
        riesgo_score += 15
    
    # Factor 3: Nivel actual
    if nivel_actual == 'C':
        riesgo_score += 30
    elif nivel_actual == 'B':
        riesgo_score += 15
    
    # Clasificar riesgo
    if riesgo_score >= 70:
        return {
            'nivel': '🔴 ALTO',
            'probabilidad': f'{min(riesgo_score, 95)}%',
            'urgencia': 'CRÍTICO',
            'color': 'error',
            'accion': 'Intervención inmediata requerida'
        }
    elif riesgo_score >= 40:
        return {
            'nivel': '🟡 MEDIO',
            'probabilidad': f'{riesgo_score}%',
            'urgencia': 'MODERADO',
            'color': 'warning',
            'accion': 'Monitoreo cercano y reforzamiento'
        }
    else:
        return {
            'nivel': '🟢 BAJO',
            'probabilidad': f'{riesgo_score}%',
            'urgencia': 'PREVENTIVO',
            'color': 'success',
            'accion': 'Mantener estrategias actuales'
        }

def generar_recomendaciones_personalizadas(
    estudiante_data: pd.Series,
    areas_debiles: List[str],
    riesgo: Dict
) -> str:
    """
    Genera recomendaciones pedagógicas personalizadas según MINEDU Arequipa 2025
    
    Args:
        estudiante_data: Serie con datos del estudiante
        areas_debiles: Lista de áreas con bajo rendimiento
        riesgo: Diccionario con nivel de riesgo
        
    Returns:
        Texto con recomendaciones en formato Markdown
    """
    promedio = estudiante_data.get('PROMEDIO', 0)
    nivel = estudiante_data.get('CALIFICACION_LETRA', 'C')
    
    recomendaciones = [
        f"### 📋 Plan de Acción Personalizado - Nivel de Riesgo: {riesgo['nivel']}\n"
    ]
    
    # Recomendaciones según nivel de riesgo
    if riesgo['nivel'] == '🔴 ALTO':
        recomendaciones.append("""
#### 🚨 Acciones Urgentes (Implementar en 72 horas):

1. **Reunión Inmediata:**
   - Convocar a padres/apoderados urgentemente
   - Establecer compromiso escrito de apoyo familiar
   - Horario de estudio supervisado en casa

2. **Plan de Reforzamiento Intensivo:**
   - **Tutorías personalizadas:** 4 sesiones/semana (45 min c/u)
   - **Horario extracurricular:** Lunes a Jueves 3:00-3:45 PM
   - **Material adaptado:** Fichas con nivel de complejidad gradual
   - **Evaluaciones semanales:** Monitoreo de avances cada viernes

3. **Apoyo Especializado:**
   - Derivación a psicología educativa si es necesario
   - Evaluación de posibles barreras de aprendizaje
   - Coordinación con SAANEE si corresponde
        """)
    
    elif riesgo['nivel'] == '🟡 MEDIO':
        recomendaciones.append("""
#### ⚠️ Acciones Preventivas (Implementar esta semana):

1. **Reforzamiento Grupal:**
   - Sesiones grupales 2-3 veces/semana
   - Grupos de 3-5 estudiantes con necesidades similares
   - Aprendizaje colaborativo con tutor-par

2. **Estrategias en Aula:**
   - Asignación de asiento estratégico (cerca del docente)
   - Retroalimentación inmediata en clase
   - Tareas diferenciadas con apoyo visual

3. **Comunicación con Familia:**
   - Reunión quincenal con apoderados
   - Reporte semanal de avances por WhatsApp/agenda
   - Estrategias de apoyo en casa
        """)
    
    else:  # Riesgo BAJO
        recomendaciones.append("""
#### ✅ Estrategias de Consolidación:

1. **Mantenimiento del Rendimiento:**
   - Monitoreo preventivo mensual
   - Desafíos académicos progresivos
   - Proyectos de profundización

2. **Desarrollo de Autonomía:**
   - Técnicas de estudio independiente
   - Gestión del tiempo académico
   - Autoevaluación y metacognición

3. **Rol de Tutor-Par:**
   - Asignar como mentor de compañeros con dificultades
   - Participación en proyectos de liderazgo académico
        """)
    
    # Recomendaciones por áreas débiles
    if areas_debiles:
        recomendaciones.append(f"\n#### 📚 Reforzamiento por Áreas Críticas:\n")
        
        estrategias_por_area = {
            'MATEMÁTICA': """
**Matemática:**
- Uso de material concreto (regletas, ábacos, geoplanos)
- Resolución de problemas contextualizados de la vida diaria
- Ejercicios graduados de menor a mayor complejidad
- Plataformas digitales: Khan Academy, Mathway (con supervisión)
- Reforzar operaciones básicas si es necesario
            """,
            'COMUNICACIÓN': """
**Comunicación:**
- Lecturas cortas con comprensión guiada (10-15 min diarios)
- Organizadores visuales (mapas mentales, esquemas)
- Producción de textos con estructura scaffold
- Diccionario personalizado de palabras nuevas
- Juegos de palabras y crucigramas adaptados
            """,
            'CIENCIA Y TECNOLOGÍA': """
**Ciencia y Tecnología:**
- Experimentos simples con materiales caseros
- Videos educativos cortos (5-7 min)
- Método científico aplicado a situaciones cotidianas
- Cuaderno de observaciones y dibujos científicos
- Visitas virtuales a museos y laboratorios
            """,
            'CIENCIAS SOCIALES': """
**Ciencias Sociales:**
- Líneas de tiempo visuales e interactivas
- Mapas conceptuales de procesos históricos
- Análisis de casos locales (historia de Arequipa)
- Proyectos de investigación de su comunidad
- Debates y foros sobre temas actuales
            """,
            'INGLÉS': """
**Inglés:**
- Aplicaciones de idiomas: Duolingo, Babbel (15 min/día)
- Canciones y videos con subtítulos
- Flashcards de vocabulario temático
- Conversaciones básicas en contextos reales
- Juegos interactivos de gramática
            """
        }
        
        for area in areas_debiles[:3]:  # Top 3 áreas
            for area_key, estrategia in estrategias_por_area.items():
                if area_key in area.upper():
                    recomendaciones.append(estrategia)
                    break
    
    # Recomendaciones según nivel MINEDU
    recomendaciones.append(f"\n#### 🎯 Estrategias según Nivel Actual ({nivel}):\n")
    recomendaciones.append(ESTRATEGIAS_MINEDU.get(nivel, ""))
    
    # Recursos adicionales
    recomendaciones.append("""
#### 🌐 Recursos Digitales Recomendados (Gratuitos):

**Plataformas Educativas:**
- [PerúEduca](https://www.perueduca.pe): Recursos oficiales MINEDU
- [Khan Academy en Español](https://es.khanacademy.org): Matemática y ciencias
- [Aprendo en Casa](https://aprendoencasa.pe): Contenidos por grado

**YouTube Educativo:**
- Matemáticas: "Matemáticas Profe Alex", "Daniel Carreón"
- Comunicación: "La Profe Pao", "Literatura y algo más"
- Ciencias: "CuriosaMente", "Quantum Fracture"

**Apps Móviles:**
- Photomath: Ayuda en matemática con cámara
- Duolingo: Inglés y otros idiomas
- Socratic by Google: Ayuda con tareas escolar
    """)
    
    # Cronograma sugerido
    recomendaciones.append("""
#### 📅 Cronograma Semanal Sugerido:

| Día | Actividad | Duración | Responsable |
|-----|-----------|----------|-------------|
| **Lunes** | Tutoría personalizada + Tarea de Matemática | 1 hora | Docente + Familia |
| **Martes** | Lectura comprensiva + Ejercicios de Comunicación | 45 min | Familia |
| **Miércoles** | Tutoría personalizada + Tarea de Ciencias | 1 hora | Docente + Familia |
| **Jueves** | Repaso general + Plataformas digitales | 45 min | Familia |
| **Viernes** | Evaluación semanal + Retroalimentación | 30 min | Docente |
| **Fin de semana** | Reforzamiento de áreas críticas + Lectura libre | 1 hora | Familia |
    """)
    
    # Referencias normativas
    recomendaciones.append("""
---

#### 📚 Referencias Normativas MINEDU:

- **RVM N° 094-2020-MINEDU:** Evaluación de Competencias
- **RVM N° 133-2020-MINEDU:** Orientaciones Retroalimentación
- **Orientaciones Pedagógicas 2025:** Estrategias de reforzamiento
- **UGEL Arequipa Sur:** Protocolos de intervención académica

#### 📞 Contacto y Seguimiento:

- **Docente:** Agendar reunión semanal de seguimiento
- **Coordinación Académica:** Monitoreo mensual
- **Padres/Apoderados:** Comunicación diaria vía agenda/WhatsApp

---

**⚠️ IMPORTANTE:** Este plan debe ser revisado y ajustado cada 2 semanas según los avances del estudiante.
    """)
    
    return "\n".join(recomendaciones)

def identificar_fortalezas_debilidades(estudiante_data: pd.Series, columnas_areas: Dict) -> Tuple[List, List]:
    """
    Identifica áreas de fortaleza y debilidad del estudiante
    
    Returns:
        (fortalezas, debilidades)
    """
    fortalezas = []
    debilidades = []
    
    for area, cols in columnas_areas.items():
        if cols:
            try:
                notas_area = [estudiante_data[col] for col in cols if col in estudiante_data.index]
                if notas_area:
                    promedio_area = sum(notas_area) / len(notas_area)
                    
                    if promedio_area >= 15:
                        fortalezas.append((area, promedio_area))
                    elif promedio_area < 13:
                        debilidades.append((area, promedio_area))
            except:
                continue
    
    # Ordenar
    fortalezas.sort(key=lambda x: x[1], reverse=True)
    debilidades.sort(key=lambda x: x[1])
    
    return fortalezas, debilidades

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def pagina_analisis_estudiantil(datos_por_hoja: Dict[str, pd.DataFrame]):
    """
    🧑‍🎓 ANÁLISIS INDIVIDUAL POR ESTUDIANTE - VERSIÓN COMPLETA
    
    Funcionalidades:
    - Búsqueda de estudiante por nombre o código
    - Historial académico completo
    - Gráficos de evolución por bimestre
    - Identificación de fortalezas y áreas de mejora
    - Predicción de riesgo académico con IA
    - Recomendaciones personalizadas MINEDU 2025
    - Exportación de informe individual profesional
    """
    
    st.title("🧑‍🎓 Análisis Individual por Estudiante")
    st.caption(f"📍 {INFO_INSTITUCION.get('nombre_ie1', 'Institución Educativa')} | Sistema de Alerta Temprana")
    
    if not datos_por_hoja:
        st.warning("⚠️ No hay datos cargados. Por favor, cargue un archivo Excel.")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. SELECTOR DE AULA
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🔍 Búsqueda de Estudiante")
    
    col_sel1, col_sel2 = st.columns([1, 2])
    
    with col_sel1:
        aula_seleccionada = st.selectbox(
            "Seleccionar aula:",
            list(datos_por_hoja.keys()),
            help="Primero seleccione el aula del estudiante"
        )
    
    df_aula = datos_por_hoja[aula_seleccionada]
    
    # Procesar datos
    with st.spinner(f"Procesando datos de {aula_seleccionada}..."):
        columnas_notas, columnas_id = obtener_columnas_notas(df_aula)
        
        if not columnas_notas:
            st.error("❌ No se encontraron columnas de notas")
            return
        
        df_procesado, columnas_num_proc = procesar_datos(df_aula, columnas_notas)
    
    # Detectar columna de nombre
    col_nombre = find_column(df_procesado, ['APELLIDOS', 'NOMBRES', 'ESTUDIANTE'])
    
    if not col_nombre:
        st.error("❌ No se pudo identificar la columna de nombres de estudiantes")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. BÚSQUEDA DE ESTUDIANTE
    # ═══════════════════════════════════════════════════════════════════════════
    
    with col_sel2:
        termino_busqueda = st.text_input(
            "Buscar estudiante por nombre:",
            placeholder="Escriba el nombre o apellido del estudiante...",
            help="Puede buscar por nombre completo o parcial"
        )
    
    # Selector alternativo por lista
    col_list1, col_list2 = st.columns([2, 1])
    
    with col_list1:
        estudiante_seleccionado = st.selectbox(
            "O seleccione de la lista:",
            options=["-- Seleccione un estudiante --"] + sorted(df_procesado[col_nombre].unique().tolist()),
            help="Lista completa de estudiantes en el aula"
        )
    
    # Determinar estudiante a mostrar
    if estudiante_seleccionado != "-- Seleccione un estudiante --":
        df_resultado = df_procesado[df_procesado[col_nombre] == estudiante_seleccionado]
    elif termino_busqueda:
        df_resultado = buscar_estudiante(df_procesado, termino_busqueda, col_nombre)
    else:
        df_resultado = pd.DataFrame()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. MOSTRAR RESULTADOS DE BÚSQUEDA
    # ═══════════════════════════════════════════════════════════════════════════
    
    if df_resultado.empty:
        st.info("👆 Seleccione un estudiante de la lista o búsquelo por nombre para ver su análisis completo.")
        
        # Mostrar estadísticas generales del aula mientras
        with st.expander("📊 Ver estadísticas generales del aula"):
            col_est1, col_est2, col_est3, col_est4 = st.columns(4)
            col_est1.metric("Total Estudiantes", len(df_procesado))
            col_est2.metric("Promedio del Aula", f"{df_procesado['PROMEDIO'].mean():.2f}")
            col_est3.metric("Aprobados", (df_procesado['ESTADO'] == 'Aprobado').sum())
            col_est4.metric("En Riesgo", (df_procesado['CALIFICACION_LETRA'] == 'C').sum())
        
        return
    
    if len(df_resultado) > 1:
        st.warning(f"⚠️ Se encontraron {len(df_resultado)} estudiantes. Refine su búsqueda:")
        st.dataframe(df_resultado[[col_nombre, 'PROMEDIO', 'CALIFICACION_LETRA']], use_container_width=True)
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. PERFIL COMPLETO DEL ESTUDIANTE
    # ═══════════════════════════════════════════════════════════════════════════
    
    estudiante = df_resultado.iloc[0]
    nombre_estudiante = estudiante[col_nombre]
    promedio_est = estudiante['PROMEDIO']
    nivel_est = estudiante['CALIFICACION_LETRA']
    estado_est = estudiante['ESTADO']
    
    st.markdown("---")
    st.markdown(f"## 👤 Perfil Académico: **{nombre_estudiante}**")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5. TARJETAS DE INFORMACIÓN PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <p style='margin: 0; font-size: 12px;'>Aula</p>
            <h3 style='margin: 5px 0;'>{aula_seleccionada}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        color_promedio = '#06D6A0' if promedio_est >= 14 else '#FFD166' if promedio_est >= 11 else '#FF6B6B'
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color_promedio} 0%, {color_promedio}dd 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <p style='margin: 0; font-size: 12px;'>Promedio</p>
            <h3 style='margin: 5px 0;'>{promedio_est:.2f}/20</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info3:
        emoji_nivel = ESCALA_CALIFICACIONES[nivel_est]['emoji']
        color_nivel = ESCALA_CALIFICACIONES[nivel_est]['color']
        st.markdown(f"""
        <div style='background: {color_nivel}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <p style='margin: 0; font-size: 12px;'>Nivel</p>
            <h3 style='margin: 5px 0;'>{emoji_nivel} {nivel_est}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info4:
        color_estado = '#06D6A0' if estado_est == 'Aprobado' else '#FF6B6B'
        st.markdown(f"""
        <div style='background: {color_estado}; padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <p style='margin: 0; font-size: 12px;'>Estado</p>
            <h3 style='margin: 5px 0;'>{estado_est}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6. PREDICCIÓN DE RIESGO ACADÉMICO
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🎯 Análisis de Riesgo Académico (Sistema de Alerta Temprana)")
    
    # Calcular tendencia (simulada por ahora, idealmente con datos de múltiples bimestres)
    tendencia = calcular_tendencia([promedio_est])  # Por ahora solo un valor
    
    # Predecir riesgo
    riesgo = predecir_riesgo_academico(promedio_est, tendencia, nivel_est)
    
    col_riesgo1, col_riesgo2 = st.columns([1, 2])
    
    with col_riesgo1:
        if riesgo['color'] == 'error':
            st.error(f"""
            **Nivel de Riesgo:** {riesgo['nivel']}
            
            **Probabilidad de Desaprobación:** {riesgo['probabilidad']}
            
            **Urgencia:** {riesgo['urgencia']}
            
            **Acción:** {riesgo['accion']}
            """)
        elif riesgo['color'] == 'warning':
            st.warning(f"""
            **Nivel de Riesgo:** {riesgo['nivel']}
            
            **Probabilidad de Riesgo:** {riesgo['probabilidad']}
            
            **Urgencia:** {riesgo['urgencia']}
            
            **Acción:** {riesgo['accion']}
            """)
        else:
            st.success(f"""
            **Nivel de Riesgo:** {riesgo['nivel']}
            
            **Probabilidad de Riesgo:** {riesgo['probabilidad']}
            
            **Urgencia:** {riesgo['urgencia']}
            
            **Acción:** {riesgo['accion']}
            """)
    
    with col_riesgo2:
        # Gráfico de gauge para riesgo
        probabilidad_num = float(riesgo['probabilidad'].replace('%', ''))
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probabilidad_num,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilidad de Desaprobación", 'font': {'size': 16}},
            delta={'reference': 30, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#06D6A0"},
                    {'range': [30, 60], 'color': "#FFD166"},
                    {'range': [60, 100], 'color': "#FF6B6B"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 7. ANÁLISIS POR ÁREAS - FORTALEZAS Y DEBILIDADES
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📚 Análisis por Áreas Curriculares")
    
    areas_map = procesar_datos_por_area(df_procesado, columnas_num_proc)
    
    # Calcular promedios por área para el estudiante
    promedios_areas = {}
    for area, cols in areas_map.items():
        if cols:
            try:
                notas = [estudiante[col] for col in cols if col in estudiante.index and pd.notna(estudiante[col])]
                if notas:
                    promedios_areas[area] = sum(notas) / len(notas)
            except:
                continue
    
    if promedios_areas:
        df_areas_est = pd.DataFrame(
            list(promedios_areas.items()),
            columns=['Área', 'Promedio']
        ).sort_values('Promedio', ascending=False)
        
        # Identificar fortalezas y debilidades
        fortalezas, debilidades = identificar_fortalezas_debilidades(estudiante, areas_map)
        
        col_areas1, col_areas2 = st.columns([1, 2])
        
        with col_areas1:
            st.markdown("#### ⭐ Fortalezas")
            if fortalezas:
                for area, promedio in fortalezas[:3]:
                    st.success(f"**{area}**: {promedio:.2f}")
            else:
                st.info("Desarrollar fortalezas específicas")
            
            st.markdown("#### ⚠️ Áreas de Mejora")
            if debilidades:
                for area, promedio in debilidades[:3]:
                    st.error(f"**{area}**: {promedio:.2f}")
            else:
                st.success("Sin áreas críticas detectadas")
        
        with col_areas2:
            # Gráfico de barras horizontal
            fig_areas = px.bar(
                df_areas_est,
                x='Promedio',
                y='Área',
                orientation='h',
                title=f'Rendimiento por Área - {nombre_estudiante}',
                color='Promedio',
                color_continuous_scale='RdYlGn',
                range_color=[0, 20],
                text='Promedio'
            )
            fig_areas.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_areas.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_areas, use_container_width=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 8. RECOMENDACIONES PERSONALIZADAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 💡 Plan de Acción Personalizado MINEDU 2025")
    
    areas_debiles = [area for area, _ in debilidades]
    recomendaciones_texto = generar_recomendaciones_personalizadas(
        estudiante,
        areas_debiles,
        riesgo
    )
    
    st.markdown(recomendaciones_texto)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 9. HISTORIAL ACADÉMICO (SIMULADO - PREPARADO PARA MÚLTIPLES BIMESTRES)
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📈 Historial Académico y Evolución")
    
    # Por ahora solo tenemos I Bimestre, pero el código está listo para más
    st.info("""
    **📊 Evolución por Bimestre** (Disponible cuando se carguen más bimestres)
    
    El sistema está preparado para mostrar:
    - Gráfico de evolución del promedio general
    - Comparación de promedios por área entre bimestres
    - Tendencias de mejora o declive
    - Predicciones para próximos bimestres
    
    **Actualmente:** Solo I Bimestre disponible
    """)
    
    # Preparar para futuro
    with st.expander("🔮 Vista previa: Gráfico de evolución (Simulación)"):
        # Simular datos de evolución
        bimestres = ['I Bim', 'II Bim (proyectado)', 'III Bim (proyectado)', 'IV Bim (proyectado)']
        promedios_sim = [promedio_est, promedio_est + 0.5, promedio_est + 1, promedio_est + 1.2]
        
        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(
            x=bimestres,
            y=promedios_sim,
            mode='lines+markers+text',
            name='Promedio',
            line=dict(color='#667eea', width=3),
            marker=dict(size=12),
            text=[f'{p:.2f}' for p in promedios_sim],
            textposition='top center'
        ))
        
        # Línea de aprobación
        fig_evol.add_hline(y=11, line_dash="dash", line_color="orange", 
                          annotation_text="Línea de Aprobación")
        
        fig_evol.update_layout(
            title='Evolución del Promedio General (Proyección)',
            xaxis_title='Bimestre',
            yaxis_title='Promedio',
            height=400,
            yaxis=dict(range=[0, 20])
        )
        
        st.plotly_chart(fig_evol, use_container_width=True)
        st.caption("*Datos proyectados basados en tendencia actual*")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 10. EXPORTACIÓN DE INFORME INDIVIDUAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📥 Exportar Informe Individual")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Crear DataFrame con información del estudiante
        df_informe = pd.DataFrame({
            'Estudiante': [nombre_estudiante],
            'Aula': [aula_seleccionada],
            'Promedio': [promedio_est],
            'Nivel': [nivel_est],
            'Estado': [estado_est],
            'Riesgo': [riesgo['nivel']],
            'Probabilidad_Riesgo': [riesgo['probabilidad']],
            'Urgencia': [riesgo['urgencia']]
        })
        
        # Agregar promedios por área
        for area, prom in promedios_areas.items():
            df_informe[f'Promedio_{area.replace(" ", "_")}'] = [prom]
        
        excel_informe = df_to_excel_bytes(df_informe, f"Informe_{nombre_estudiante.replace(' ', '_')}")
        
        st.download_button(
            label="📊 Descargar Informe Excel",
            data=excel_informe,
            file_name=f"informe_{nombre_estudiante.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_exp2:
        # Exportar recomendaciones en texto
        recomendaciones_txt = f"""
INFORME ACADÉMICO INDIVIDUAL
{'='*50}

ESTUDIANTE: {nombre_estudiante}
AULA: {aula_seleccionada}
FECHA: {datetime.now().strftime('%d/%m/%Y')}

{'='*50}
RESUMEN ACADÉMICO
{'='*50}

Promedio General: {promedio_est:.2f}/20
Nivel de Logro: {nivel_est} - {ESCALA_CALIFICACIONES[nivel_est]['desc']}
Estado: {estado_est}

{'='*50}
ANÁLISIS DE RIESGO
{'='*50}

Nivel de Riesgo: {riesgo['nivel']}
Probabilidad: {riesgo['probabilidad']}
Urgencia: {riesgo['urgencia']}
Acción Requerida: {riesgo['accion']}

{'='*50}
RECOMENDACIONES PERSONALIZADAS
{'='*50}

{recomendaciones_texto}

{'='*50}
Generado por: Sistema Académico MINEDU V5.1
I.E. {INFO_INSTITUCION.get('nombre_ie1', '')}
{'='*50}
        """
        
        st.download_button(
            label="📄 Descargar Recomendaciones TXT",
            data=recomendaciones_txt.encode('utf-8'),
            file_name=f"recomendaciones_{nombre_estudiante.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_exp3:
        if st.button("✉️ Enviar Informe a Padres", use_container_width=True):
            st.info("""
            **📧 Funcionalidad en desarrollo**
            
            Próximamente podrás:
            - Enviar informe por correo electrónico
            - Generar PDF profesional
            - Registrar en historial de comunicaciones
            - Solicitar firma digital de recepción
            """)
    
    st.success(f"✅ Análisis completo de **{nombre_estudiante}** generado exitosamente.")
