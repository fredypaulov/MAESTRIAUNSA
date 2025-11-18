# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                       PÁGINAS AUXILIARES                                  ║
║        Análisis de Priorizados, Ayuda, Exportar Reportes, etc.           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from constantes import INFO_INSTITUCION, EQUIVALENCIAS_NOTAS
from utils import find_column, df_to_excel_bytes

# ═════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE ESTUDIANTES PRIORIZADOS
# ═════════════════════════════════════════════════════════════════════════════

# def pagina_analisis_priorizados(datos_raw: Optional[Dict[str, pd.DataFrame]]):
#    """Página de análisis de estudiantes priorizados"""
#    st.title("🎯 Análisis de Estudiantes Priorizados")
#    st.caption("Identificación de estudiantes que requieren reforzamiento académico")
    
#    if not datos_raw:
#        st.warning("⚠️ No hay datos cargados.")
#        return
    
#    st.info("🚧 **Módulo en desarrollo avanzado**")
#    st.markdown("""
    ### Funcionalidades Planificadas:
#    - Detección automática de estudiantes en riesgo
#    - Análisis por área de aprendizaje
#    - Mapas de calor de rendimiento
#    - Recomendaciones pedagógicas personalizadas
#    - Exportación de listas para intervención
#    """)

# ═════════════════════════════════════════════════════════════════════════════
# ANÁLISIS INDIVIDUAL POR ESTUDIANTE
# ═════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# 🛠️ Paso 3: Actualización de Página Estudiantil
# ----------------------------------------------------------------------------
# La función `pagina_analisis_estudiantil()` ha sido migrada a su propio módulo
# dedicado (`vista_estudiantil.py`) como parte del refactor modular del sistema.
# 
# Por ello, esta versión placeholder ha sido comentada/eliminada para evitar 
# duplicidad de funciones y asegurar que la versión activa sea la más reciente.
# 
# 🎯 Cambios realizados:
# - ✅ Se creó archivo: vista_estudiantil.py con la versión completa.
# - ✅ `app_dashboard.py` ahora importa desde `vista_estudiantil`.
# - ❌ Esta función ha sido comentada aquí para futura referencia o remoción final.
# ════════════════════════════════════════════════════════════════════════════



# def pagina_analisis_estudiantil(datos_por_hoja: Dict):
#    """Análisis individualizado por estudiante"""
#    st.title("🧑‍🎓 Análisis Individual por Estudiante")
#    st.caption("Perfil académico detallado")
    
#    if not datos_por_hoja:
#        st.warning("⚠️ No hay datos cargados")
#        return
 
#    st.info("🚧 **Módulo en desarrollo**")
#    st.markdown("""
#    ### Funcionalidades Planificadas:
#    - Búsqueda de estudiante por nombre o código
#    - Historial académico completo
#    - Gráficos de evolución por bimestre
#    - Fortalezas y áreas de mejora
#    - Recomendaciones personalizadas
#    - Exportación de informe individual
#    """)

# ═════════════════════════════════════════════════════════════════════════════
# MODELO PREDICTIVO
# ═════════════════════════════════════════════════════════════════════════════

# def pagina_modelo_predictivo(datos_raw: Dict):
#    """Modelo predictivo con Machine Learning"""
#    st.title("🤖 Modelo Predictivo de Rendimiento Académico")
#    st.caption("Predicciones basadas en Machine Learning")
    
#    st.info("🚧 **Módulo Predictivo en desarrollo**")
#    st.markdown("""
#    ### Funcionalidades Planificadas:
#    - Predicción de rendimiento futuro
#    - Identificación temprana de riesgo
#    - Factores que influyen en el rendimiento
#    - Recomendaciones automatizadas
#    - Modelos: CatBoost, XGBoost, Random Forest
    
#    **Requisitos:**
#    - Datos de múltiples bimestres/periodos
#   - Bibliotecas: `catboost`, `xgboost`, `scikit-learn`
#    """)

# ═════════════════════════════════════════════════════════════════════════════
# EXPORTAR REPORTES
# ═════════════════════════════════════════════════════════════════════════════

# def pagina_exportar_reportes(datos_por_hoja: Dict):
#    """Centro de exportación de reportes"""
#    st.title("📄 Exportar Reportes Institucionales")
#    st.caption("Descarga de reportes en múltiples formatos")
    
#    if not datos_por_hoja:
#        st.warning("⚠️ No hay datos cargados")
#        return
    
#    st.info("🚧 **Centro de Reportes en construcción**")
#    st.markdown("""
    ### Tipos de Reportes Disponibles:
    
    #### 📊 Reportes Académicos:
#    - Boletas de notas por estudiante
#    - Actas de evaluación por grado
#    - Informes de progreso bimestral
#    - Certificados de estudios
    
    #### 📈 Reportes Estadísticos:
#    - Análisis comparativo entre aulas
#    - Tendencias de rendimiento institucional
#    - Dashboards ejecutivos
#    - Reportes para UGEL/MINEDU
    
    #### 🎯 Reportes de Intervención:
#    - Listas de priorizados
#    - Planes de reforzamiento
#    - Seguimiento de tutorías
#    - Comunicados a padres
    
#    **Formatos:** Excel, PDF, CSV, Word
#   """)

# ═════════════════════════════════════════════════════════════════════════════
# AYUDA Y SOPORTE
# ═════════════════════════════════════════════════════════════════════════════

def pagina_ayuda():
    """Página de ayuda y soporte"""
    st.title("❓ Ayuda y Soporte del Sistema")
    st.caption("Guía de uso y referencias normativas")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Guía de Uso",
        "🎓 Normativa MINEDU",
        "🛠️ Solución de Problemas",
        "📞 Contacto"
    ])
    
    with tab1:
        st.markdown("""
        ### 📖 Guía Rápida de Uso
        
        #### 1. Carga de Datos
        1. Clic en **"Browse files"** en la barra lateral
        2. Seleccione archivo Excel (.xlsx o .xls)
        3. Sistema detecta automáticamente hojas y columnas
        
        #### 2. Navegación por Vistas
        - **🏠 Inicio:** Información general
        - **👨‍🏫 Vista Director:** Análisis global institucional
        - **👩‍🏫 Vista Docente:** Análisis detallado por aula
        - **🎯 Análisis Priorizados:** Estudiantes en reforzamiento
        
        #### 3. Interpretación de Niveles
        - **AD (18-20):** Logro Destacado
        - **A (15-17):** Logro Esperado
        - **B (11-14):** En Proceso
        - **C (0-10):** En Inicio
        """)
    
    with tab2:
        st.markdown("""
        ### 🎓 Normativa MINEDU de Referencia
        
        #### Documentos Normativos:
        1. **RVM N° 094-2020-MINEDU**
           - Evaluación de Competencias
           - Escala de calificación
        
        2. **RVM N° 334-2021-MINEDU**
           - Disposiciones año escolar
           - Orientaciones pedagógicas
        
        3. **Currículo Nacional**
           - Enfoque por competencias
           - Estándares de aprendizaje
        
        #### Enlaces Útiles:
        - [Portal MINEDU](https://www.minedu.gob.pe)
        - [PerúEduca](https://www.perueduca.pe)
        - [SIAGIE](http://siagie.minedu.gob.pe)
        """)
    
    with tab3:
        st.markdown("""
        ### 🛠️ Solución de Problemas
        
        #### ❌ Error al cargar archivo
        **Soluciones:**
        1. Verificar formato (.xlsx o .xls)
        2. Asegurar que hojas contengan datos
        3. Verificar columnas con nombres y notas
        4. Revisar formato de notas (AD/A/B/C o 0-20)
        
        #### ⚠️ No se detectan columnas
        **Soluciones:**
        1. Columnas deben tener palabras clave
        2. Verificar sin filas vacías antes del encabezado
        3. Revisar formato de celdas
        
        #### 🔄 Sistema lento
        **Soluciones:**
        1. Cerrar otras pestañas
        2. Actualizar página (F5)
        3. Limpiar caché
        4. Usar navegadores modernos
        """)
    
    with tab4:
        st.markdown(f"""
        ### 📞 Información de Contacto
        
        #### Institución:
        **{INFO_INSTITUCION['nombre_ie1']}**
        - 📍 {INFO_INSTITUCION['ubicacion']}
        - 🔢 Código: {INFO_INSTITUCION['codigo']}
        - 🏫 UGEL: {INFO_INSTITUCION['ugel']}
        
        #### Soporte Técnico:
        - 💻 **Versión:** {INFO_INSTITUCION['version']}
        - 📧 **Email:** ievinvasecundaria@gmail.com
        - 📱 **Teléfono:** (054) 344259
        
        ---
        
        ### 🙏 Créditos
        **Desarrollado por:** Alan Turing 🧠  
        **Optimizado con:** Arquitectura Modular Python  
        **Tecnologías:** Streamlit, Plotly, Pandas, scikit-learn
        """)
