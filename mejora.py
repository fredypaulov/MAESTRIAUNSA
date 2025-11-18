
# -*- coding: utf-8 -*-
"""
====================================================================
🎓 SISTEMA PREDICTIVO DE REFORZAMIENTO ESCOLAR - v2.0
====================================================================
Aplicación Multi-Página para Directores y Docentes
Basado en los requisitos de MINEDU y del I.E. Victor Andres Belaunde
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import warnings
from datetime import datetime

# Ignorar advertencias comunes
warnings.filterwarnings('ignore')

# Intentar importar CatBoost, si falla, se deshabilitará el ML
try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    st.error("Advertencia: La librería 'catboost' no está instalada. El modelado predictivo estará deshabilitado. Instala con: pip install catboost")

# ============================================================
# CONFIGURACIÓN CENTRAL DEL SISTEMA
# ============================================================

# Escala de calificaciones EXACTA proporcionada por el usuario
ESCALA_CALIFICACIONES = {
    "AD": {"min": 18, "max": 20, "num": 19, "desc": "Logro Destacado"},
    "A":  {"min": 15, "max": 17, "num": 16, "desc": "Logro Esperado"},
    "B":  {"min": 11, "max": 14, "num": 12, "desc": "En Proceso"},
    "C":  {"min": 0,  "max": 10, "num": 8,  "desc": "En Inicio"}
}

# Base de conocimiento para recomendaciones (basada en enlaces MINEDU)
ESTRATEGIAS_MINEDU = {
    "C": "🚨 **Reforzamiento Urgente**: Requiere un plan de tutoría individualizado. Foco en competencias básicas. (Ref: MINEDU PEI 2024). Contactar a padres de familia.",
    "B": "⚠️ **Acompañamiento Requerido**: Estudiante 'En Proceso'. Necesita material didáctico complementario, fomentar trabajo colaborativo y seguimiento semanal. (Ref: MINEDU).",
    "A": "✅ **Logro Esperado**: Consolidar aprendizaje. Asignar proyectos de aplicación práctica para afianzar competencias.",
    "AD": "🌟 **Logro Destacado**: Fomentar investigación, mentoría a compañeros y preparación para olimpiadas académicas. (Ref: PEI 40079)."
}

# ============================================================
# FUNCIONES DE UTILIDAD (DATOS)
# ============================================================

@st.cache_data
def cargar_excel(archivo_subido):
    """
    Carga todas las hojas de un archivo Excel y devuelve un diccionario de DataFrames.
    """
    try:
        xls = pd.ExcelFile(archivo_subido)
        nombres_hojas = xls.sheet_names
        if not nombres_hojas:
            st.error("El archivo Excel no tiene hojas.")
            return None, []
        
        # Cargar todas las hojas, detectando el encabezado correcto
        datos_por_hoja = {}
        for hoja in nombres_hojas:
            df_raw = pd.read_excel(xls, sheet_name=hoja, header=None)
            fila_header = detectar_fila_encabezado(df_raw)
            
            if fila_header is not None:
                # Volver a cargar con el encabezado correcto
                df_hoja = pd.read_excel(xls, sheet_name=hoja, header=fila_header)
                df_hoja = limpiar_dataframe(df_hoja)
                if not df_hoja.empty:
                    datos_por_hoja[hoja] = df_hoja
            else:
                st.warning(f"No se pudo encontrar un encabezado válido en la hoja: '{hoja}'. Se omitirá.")
        
        if not datos_por_hoja:
            st.error("No se pudo cargar ninguna hoja con datos válidos.")
            return None, []
            
        return datos_por_hoja, list(datos_por_hoja.keys())
    
    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")
        return None, []

def detectar_fila_encabezado(df_raw, palabras_clave=['APELLIDOS', 'NOMBRE', 'ESTUDIANTE']):
    """
    Detecta automáticamente la fila que contiene el encabezado.
    """
    for i in range(min(10, len(df_raw))):
        fila_str = ' '.join(str(x).upper() for x in df_raw.iloc[i] if pd.notna(x))
        if any(clave in fila_str for clave in palabras_clave):
            return i
    return None # Si no lo encuentra, asume que es la primera (header=0) al re-leer

def limpiar_dataframe(df):
    """
    Limpia un DataFrame: elimina columnas/filas vacías y normaliza nombres.
    """
    # Renombrar columnas duplicadas si existen
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}.{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    
    # Eliminar columnas que sean completamente NaN o se llamen 'Unnamed'
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    df = df.dropna(axis=1, how='all')
    
    # Eliminar filas donde el identificador principal (usualmente la primera o segunda col) es NaN
    if len(df.columns) > 1:
        col_identificadora = df.columns[1] # Asumimos que la segunda col tiene nombres
        df = df.dropna(subset=[col_identificadora])
        
    df = df.reset_index(drop=True)
    return df

def obtener_columnas_notas(df):
    """
    Detecta columnas que probablemente sean notas (numéricas o A/B/C/AD).
    """
    columnas_notas = []
    columnas_id = []
    
    for col in df.columns:
        col_str = str(col).upper()
        # Ignorar columnas de identificación
        if any(kw in col_str for kw in ['ESTUDIANTE', 'NOMBRE', 'APELLIDO', 'GRADO', 'SECCION']):
            columnas_id.append(col)
            continue
        
        # Ignorar columnas administrativas obvias
        if any(kw in col_str for kw in ['CODIGO', 'DNI', 'ID', 'PROMEDIO', 'OBSERVACION']):
            continue

        # Analizar una muestra de la columna
        muestra = df[col].dropna().sample(min(20, len(df[col].dropna())))
        
        if muestra.empty:
            continue

        # Criterio 1: Mayoría son numéricas entre 0 y 20
        try:
            muestra_num = pd.to_numeric(muestra, errors='coerce').dropna()
            if len(muestra_num) / len(muestra) > 0.8 and muestra_num.min() >= 0 and muestra_num.max() <= 20:
                columnas_notas.append(col)
                continue
        except Exception:
            pass # No es numérico

        # Criterio 2: Mayoría son A/B/C/AD
        try:
            muestra_str = muestra.astype(str).str.upper().str.strip()
            conteo_letras = muestra_str.isin(['A', 'B', 'C', 'AD']).sum()
            if conteo_letras / len(muestra) > 0.7:
                columnas_notas.append(col)
                continue
        except Exception:
            pass
            
    return columnas_notas, columnas_id

@st.cache_data
def procesar_datos(df, columnas_notas):
    """
    Calcula promedios y convierte notas.
    """
    df_proc = df.copy()
    
    # Convertir notas a numérico
    for col in columnas_notas:
        col_num_nombre = f"{col}_num"
        
        # Primero intentar conversión numérica directa
        df_proc[col_num_nombre] = pd.to_numeric(df_proc[col], errors='coerce')
        
        # Para los que fallaron (ej. 'A', 'B', 'C'), usar el mapeo
        mapeo_letras = {
            'AD': ESCALA_CALIFICACIONES['AD']['num'],
            'A': ESCALA_CALIFICACIONES['A']['num'],
            'B': ESCALA_CALIFICACIONES['B']['num'],
            'C': ESCALA_CALIFICACIONES['C']['num']
        }
        
        # Aplicar mapeo donde la conversión numérica falló
        df_proc[col_num_nombre] = df_proc[col_num_nombre].fillna(
            df_proc[col].astype(str).str.upper().str.strip().map(mapeo_letras)
        )
    
    columnas_num_proc = [f"{col}_num" for col in columnas_notas]
    
    # Imputar NaN restantes con la mediana (ej. 8)
    for col in columnas_num_proc:
        df_proc[col] = df_proc[col].fillna(ESCALA_CALIFICACIONES['C']['num'])
        
    # Calcular métricas
    df_proc['PROMEDIO'] = df_proc[columnas_num_proc].mean(axis=1).round(2)
    df_proc['CALIFICACION_LETRA'] = df_proc['PROMEDIO'].apply(convertir_promedio_a_letra)
    df_proc['ESTADO'] = df_proc['CALIFICACION_LETRA'].apply(lambda x: 'Aprobado' if x in ['AD', 'A', 'B'] else 'Desaprobado')
    
    return df_proc, columnas_num_proc

def convertir_promedio_a_letra(promedio):
    """
    Usa la ESCALA_CALIFICACIONES definida por el usuario para convertir un promedio numérico.
    """
    for letra, rango in ESCALA_CALIFICACIONES.items():
        if rango['min'] <= promedio <= rango['max']:
            return letra
    return "C" # Default si está fuera de rango (aunque 0-20 debería cubrir todo)

# ============================================================
# FUNCIONES DE ML (Simplificadas para Streamlit)
# ============================================================

@st.cache_resource
def entrenar_modelo_catboost(_df_entrenamiento, features, target):
    """
    Entrena un modelo CatBoost y lo cachea.
    """
    if not HAS_CATBOOST:
        return None
        
    try:
        modelo = CatBoostClassifier(iterations=100,
                                  learning_rate=0.1,
                                  depth=6,
                                  loss_function='Logloss',
                                  verbose=False,
                                  random_seed=42)
        
        modelo.fit(_df_entrenamiento[features], _df_entrenamiento[target])
        return modelo
    except Exception as e:
        st.warning(f"No se pudo entrenar el modelo CatBoost: {e}")
        return None

def generar_observacion_mejorada(promedio, nombre_estudiante="el estudiante"):
    """
    Genera observación pedagógica usando la nueva escala y contexto MINEDU.
    """
    letra = convertir_promedio_a_letra(promedio)
    descripcion = ESCALA_CALIFICACIONES[letra]['desc']
    estrategia = ESTRATEGIAS_MINEDU[letra]
    
    observacion = f"**Estudiante:** {nombre_estudiante}\n"
    observacion += f"**Promedio:** {promedio:.2f} | **Nivel:** {letra} ({descripcion})\n"
    observacion += f"**Observación Pedagógica:**\n{estrategia}"
    
    return observacion, letra

# ============================================================
# FUNCIONES DE PÁGINA
# ============================================================

def mostrar_logo(nombre_colegio="I.E. Victor Andres Belaunde"):
    """Muestra el logo y nombre del colegio en la barra lateral."""
    # Reemplaza esta URL con la URL real del logo de tu colegio
    logo_url = "https://placehold.co/150x150/003366/FFFFFF?text=LOGO" 
    st.sidebar.image(logo_url, width=100)
    st.sidebar.title(nombre_colegio)

def pagina_inicio():
    st.title("🎓 Sistema de Análisis y Reforzamiento Académico")
    st.markdown("Bienvenido al panel de control para Directores y Docentes. Por favor, cargue su archivo de calificaciones (Excel) usando el menú de la izquierda para comenzar.")
    
    st.info("""
    **Instrucciones:**
    1.  Haga clic en el botón `Browse files` en la barra lateral izquierda.
    2.  Seleccione su archivo Excel (`.xlsx` o `.xls`) que contenga las notas.
    3.  Una vez cargado, las opciones de navegación aparecerán en la barra lateral.
    4.  Explore las diferentes vistas: `Vista Director`, `Vista Docente`, etc.
    """)

def pagina_vista_director(datos_por_hoja):
    st.title("👨‍🏫 Vista Director: Análisis Global del Colegio")
    st.markdown("Esta vista consolida la información de *todas* las aulas (hojas) en el archivo.")
    
    try:
        # Combinar todos los DataFrames de todas las hojas
        df_global = pd.concat(datos_por_hoja.values(), ignore_index=True)
        
        columnas_notas, columnas_id = obtener_columnas_notas(df_global)
        
        if not columnas_notas:
            st.warning("No se pudieron detectar columnas de notas en el archivo.")
            return
            
        df_procesado, _ = procesar_datos(df_global, columnas_notas)
        
        # KPIs Globales
        total_estudiantes = len(df_procesado)
        promedio_global = df_procesado['PROMEDIO'].mean()
        tasa_aprobacion = (df_procesado['ESTADO'] == 'Aprobado').mean() * 100
        
        st.markdown("### Indicadores Clave (KPIs) del Colegio")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Estudiantes", f"{total_estudiantes} 🎓")
        col2.metric("Promedio General", f"{promedio_global:.2f} 📊")
        col3.metric("Tasa de Aprobación", f"{tasa_aprobacion:.1f}% ✅")
        
        st.markdown("---")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribución de Calificaciones (Global)")
            conteo_calificaciones = df_procesado['CALIFICACION_LETRA'].value_counts().reindex(['C', 'B', 'A', 'AD']).fillna(0)
            fig = px.bar(conteo_calificaciones, 
                         x=conteo_calificaciones.index, 
                         y=conteo_calificaciones.values,
                         labels={'x': 'Calificación', 'y': 'N° de Estudiantes'},
                         color=conteo_calificaciones.index,
                         color_discrete_map={'C': '#FF6B6B', 'B': '#FFD166', 'A': '#06D6A0', 'AD': '#118AB2'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Distribución de Promedios (Global)")
            fig = px.histogram(df_procesado, 
                               x='PROMEDIO', 
                               nbins=20,
                               labels={'x': 'Promedio', 'y': 'Frecuencia'},
                               color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar la vista de Director: {e}")
        st.dataframe(df_global.head()) # Mostrar datos crudos para depuración

def pagina_vista_docente(datos_por_hoja):
    st.title("👩‍🏫 Vista Docente: Análisis por Aula/Trimestre")
    st.markdown("Seleccione una hoja de su archivo Excel para analizarla individualmente.")
    
    if not datos_por_hoja:
        st.warning("No hay hojas cargadas. Por favor, suba un archivo.")
        return

    # Selector de hoja
    hoja_seleccionada = st.selectbox("Seleccione un Aula o Trimestre:", options=list(datos_por_hoja.keys()))
    
    if hoja_seleccionada:
        try:
            df_hoja = datos_por_hoja[hoja_seleccionada]
            columnas_notas, columnas_id = obtener_columnas_notas(df_hoja)
            
            if not columnas_notas:
                st.warning(f"No se pudieron detectar columnas de notas en la hoja '{hoja_seleccionada}'.")
                return
                
            df_procesado, _ = procesar_datos(df_hoja, columnas_notas)
            
            # KPIs por Hoja
            total_estudiantes = len(df_procesado)
            promedio_hoja = df_procesado['PROMEDIO'].mean()
            tasa_aprobacion = (df_procesado['ESTADO'] == 'Aprobado').mean() * 100
            
            st.markdown(f"### Indicadores Clave (KPIs) para: **{hoja_seleccionada}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Estudiantes", f"{total_estudiantes} 🎓")
            col2.metric("Promedio del Aula", f"{promedio_hoja:.2f} 📊")
            col3.metric("Tasa de Aprobación", f"{tasa_aprobacion:.1f}% ✅")
            
            st.markdown("---")
            
            # Gráficos
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Distribución de Calificaciones ({hoja_seleccionada})")
                conteo_calificaciones = df_procesado['CALIFICACION_LETRA'].value_counts().reindex(['C', 'B', 'A', 'AD']).fillna(0)
                fig = px.bar(conteo_calificaciones, 
                             x=conteo_calificaciones.index, 
                             y=conteo_calificaciones.values,
                             labels={'x': 'Calificación', 'y': 'N° de Estudiantes'},
                             color=conteo_calificaciones.index,
                             color_discrete_map={'C': '#FF6B6B', 'B': '#FFD166', 'A': '#06D6A0', 'AD': '#118AB2'})
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown(f"#### Distribución de Promedios ({hoja_seleccionada})")
                fig = px.histogram(df_procesado, 
                                   x='PROMEDIO', 
                                   nbins=20,
                                   labels={'x': 'Promedio', 'y': 'Frecuencia'},
                                   color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ocurrió un error al procesar la hoja '{hoja_seleccionada}': {e}")
            st.dataframe(df_hoja.head()) # Mostrar datos crudos para depuración

def pagina_analisis_estudiantil(datos_por_hoja):
    st.title("🧑‍🎓 Análisis Estudiantil y Observaciones")
    st.markdown("Análisis detallado por estudiante con recomendaciones pedagógicas.")
    
    if not datos_por_hoja:
        st.warning("No hay hojas cargadas. Por favor, suba un archivo.")
        return
        
    hoja_seleccionada = st.selectbox("Seleccione un Aula para analizar:", options=list(datos_por_hoja.keys()))
    
    if hoja_seleccionada:
        try:
            df_hoja = datos_por_hoja[hoja_seleccionada]
            columnas_notas, columnas_id = obtener_columnas_notas(df_hoja)
            
            if not columnas_id:
                st.error("No se pudo encontrar una columna de nombres de estudiantes.")
                return
            
            col_nombre = columnas_id[0] # Asumir que la primera es el nombre
            
            df_procesado, columnas_num_proc = procesar_datos(df_hoja, columnas_notas)
            
            st.markdown("### Observaciones Individuales")
            
            # Buscar un estudiante
            lista_estudiantes = ["Todos"] + list(df_procesado[col_nombre].unique())
            estudiante_seleccionado = st.selectbox("Seleccione un estudiante (o 'Todos'):", options=lista_estudiantes)
            
            df_filtrado = df_procesado
            if estudiante_seleccionado != "Todos":
                df_filtrado = df_procesado[df_procesado[col_nombre] == estudiante_seleccionado]
            
            if df_filtrado.empty:
                st.info("No se encontraron datos para el estudiante seleccionado.")
                return

            # Mostrar observaciones
            for _, estudiante in df_filtrado.iterrows():
                nombre = estudiante[col_nombre]
                promedio = estudiante['PROMEDIO']
                
                observacion, letra = generar_observacion_mejorada(promedio, nombre)
                
                if letra == 'AD':
                    st.success(observacion)
                elif letra == 'A':
                    st.info(observacion)
                elif letra == 'B':
                    st.warning(observacion)
                else: # letra == 'C'
                    st.error(observacion)
            
        except Exception as e:
            st.error(f"Ocurrió un error al generar el análisis estudiantil: {e}")

def pagina_exportar_reportes(datos_por_hoja):
    st.title("📄 Exportar Reportes Personalizados")
    st.markdown("Genere y descargue un archivo CSV o Excel con los datos procesados.")
    
    if not datos_por_hoja:
        st.warning("No hay hojas cargadas. Por favor, suba un archivo.")
        return

    hoja_seleccionada = st.selectbox("Seleccione los datos del Aula/Trimestre a exportar:", options=list(datos_por_hoja.keys()))

    if hoja_seleccionada:
        try:
            df_hoja = datos_por_hoja[hoja_seleccionada]
            columnas_notas, columnas_id = obtener_columnas_notas(df_hoja)
            
            if not columnas_notas:
                st.warning(f"No hay notas válidas en la hoja '{hoja_seleccionada}'.")
                return
            
            df_procesado, _ = procesar_datos(df_hoja, columnas_notas)
            
            # Añadir observaciones al reporte
            df_procesado['OBSERVACION_PEDAGOGICA'] = df_procesado.apply(
                lambda row: generar_observacion_mejorada(row['PROMEDIO'], row[columnas_id[0]])[0],
                axis=1
            )
            
            st.markdown(f"#### Vista Previa del Reporte para: **{hoja_seleccionada}**")
            columnas_export = columnas_id + ['PROMEDIO', 'CALIFICACION_LETRA', 'ESTADO', 'OBSERVACION_PEDAGOGICA']
            st.dataframe(df_procesado[columnas_export].head())
            
            # Convertir a CSV para descarga
            @st.cache_data
            def convertir_a_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convertir_a_csv(df_procesado[columnas_export])
            
            st.download_button(
                label="📥 Descargar Reporte como CSV",
                data=csv_data,
                file_name=f"reporte_{hoja_seleccionada.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Error al generar el reporte para exportar: {e}")

def pagina_ayuda_referencias():
    st.title("❓ Ayuda y Referencias (MINEDU/SUNEDU)")
    
    st.markdown("### Información del Colegio")
    st.info("""
    * **Institución Educativa:** 40079 VICTOR NUÑEZ VALENCIA
    * **Ubicación:** Avenida Wanders 113, distrito de Sachaca, Arequipa.
    * **Gestión:** Pública de acceso gratuito.
    * **Población:** Jóvenes de 12 a 17 años (Secundaria).
    * **Promedio por Aula:** 26 estudiantes.
    * **Turno:** Tarde.
    * **Código Modular:** 0899120
    """)
    
    st.markdown("### Referencias Pedagógicas (MINEDU)")
    st.markdown("""
    Las observaciones y estrategias de este sistema se basan en los siguientes lineamientos:
    * [Plan Estratégico Institucional (PEI) 2024-2027 del MINEDU](https://www.minedu.gob.pe/normatividad/plan_institucional/pei2024/rm_167-2024-minedu_pei_2019-2027-periodo2019-2027.pdf)
    * [Planes Estratégicos del Ministerio de Educación (gob.pe)](https://www.gob.pe/65735-ministerio-de-educacion-plan-estrategico-institucional-pei)
    """)
    
    st.markdown("### Preguntas Frecuentes (FAQ)")
    
    with st.expander("¿Cómo se calcula la 'Calificación Letra' (C, B, A, AD)?"):
        st.markdown("Se utiliza la escala oficial proporcionada:")
        st.json(ESCALA_CALIFICACIONES, expanded=False)
    
    with st.expander("¿Cómo se generan las observaciones?"):
        st.markdown("Las observaciones se generan automáticamente combinando dos factores:")
        st.markdown("1.  **La escala de calificación**: Se identifica si el estudiante está 'En Inicio' (C), 'En Proceso' (B), etc.")
        st.markdown("2.  **Base de Conocimiento (MINEDU)**: Se aplica una estrategia pedagógica recomendada para ese nivel, basada en los documentos oficiales del PEI.")
    
    with st.expander("¿Qué hago si mi archivo Excel tiene un formato diferente?"):
        st.markdown("El sistema intenta detectar automáticamente la fila de encabezado (donde están los nombres de los alumnos). Si falla, asegúrese de que su archivo tenga una fila con 'APELLIDOS', 'NOMBRE' o 'ESTUDIANTE' justo encima de la lista de alumnos.")

# ============================================================
# APLICACIÓN PRINCIPAL (MAIN)
# ============================================================

def main():
    
    mostrar_logo("I.E. Victor Andres Belaunde") # Logo solicitado
    
    st.sidebar.markdown("---")
    
    # 1. Carga de Archivo
    archivo_subido = st.sidebar.file_uploader(
        "Cargar Archivo Excel",
        type=["xlsx", "xls"],
        help="Cargue el archivo Excel con las calificaciones de los estudiantes."
    )
    
    datos_cargados = None
    nombres_hojas = []
    
    if "datos_cargados" not in st.session_state:
        st.session_state.datos_cargados = None

    if archivo_subido is not None:
        # Cargar los datos y guardarlos en el estado de la sesión
        datos_cargados, nombres_hojas = cargar_excel(archivo_subido)
        st.session_state.datos_cargados = datos_cargados
    
    # 2. Navegación
    if st.session_state.datos_cargados:
        st.sidebar.markdown("---")
        st.sidebar.header("Navegación Principal")
        
        paginas = {
            "🏠 Inicio": pagina_inicio,
            "👨‍🏫 Vista Director": pagina_vista_director,
            "👩‍🏫 Vista Docente": pagina_vista_docente,
            "🧑‍🎓 Análisis Estudiantil": pagina_analisis_estudiantil,
            "📄 Exportar Reportes": pagina_exportar_reportes,
            "❓ Ayuda y Referencias": pagina_ayuda_referencias
        }
        
        seleccion_pagina = st.sidebar.radio("Ir a:", options=list(paginas.keys()))
        
        # Pasar los datos cargados a la página seleccionada
        if seleccion_pagina == "🏠 Inicio":
            paginas[seleccion_pagina]() # Inicio no necesita datos
        elif seleccion_pagina == "❓ Ayuda y Referencias":
            paginas[seleccion_pagina]() # Ayuda no necesita datos
        else:
            paginas[seleccion_pagina](st.session_state.datos_cargados)
            
    else:
        # Si no hay archivo, solo mostrar la página de inicio
        pagina_inicio()

if __name__ == "__main__":
    main()