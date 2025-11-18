
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                 SISTEMA ACADÉMICO MINEDU V5.1                             ║
║              🎓 Arquitectura Modular Profesional                          ║
║              💻 Desarrollado por: Alan Turing 🧠                          ║
║              📅 Fecha: 17 de Noviembre, 2025                              ║
║              🏆 La Mejor Arquitectura Python para Educación               ║
╚═══════════════════════════════════════════════════════════════════════════╝

CARACTERÍSTICAS:
✅ Arquitectura modular profesional
✅ Separación de responsabilidades (SOLID)
✅ Cache inteligente para optimización
✅ Manejo robusto de errores
✅ Análisis con Machine Learning
✅ Visualizaciones profesionales con Plotly
✅ Métricas MINEDU 2024-2025
"""

import streamlit as st
import sys
from pathlib import Path

# Configuración inicial de la página (DEBE ser lo primero)
st.set_page_config(
    page_title="Sistema Académico MINEDU V5.1",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.minedu.gob.pe',
        'Report a bug': None,
        'About': "Sistema Académico V5.1 - Arquitectura Modular por Alan Turing 🧠"
    }
)

# Importaciones del proyecto
try:
    from constantes import INFO_INSTITUCION, ESCALA_CALIFICACIONES
    from procesamiento import cargar_excel
    # Importar las nuevas vistas modulares
    from vista_priorizados import pagina_analisis_priorizados
    from vista_estudiantil import pagina_analisis_estudiantil
    from vista_predictivo import pagina_modelo_predictivo
    from vista_reportes import pagina_exportar_reportes
    from vista_estudiantil import pagina_analisis_estudiantil
    from vista_predictivo import pagina_modelo_predictivo
    from vista_priorizados import pagina_analisis_priorizados
    from vista_director import pagina_vista_director
    from vista_reportes import pagina_exportar_reportes
    from vista_docente import pagina_vista_docente
    from paginas_auxiliares import (
        pagina_ayuda
    )
except ImportError as e:
    st.error(f"❌ Error al importar módulos: {e}")
    st.info("Asegúrate de que todos los archivos del proyecto estén en la misma carpeta.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ═════════════════════════════════════════════════════════════════════════════

def inicializar_session_state():
    """Inicializa variables de estado de la sesión"""
    if "datos_cargados" not in st.session_state:
        st.session_state.datos_cargados = None
    if "datos_raw" not in st.session_state:
        st.session_state.datos_raw = None
    if "nombres_hojas" not in st.session_state:
        st.session_state.nombres_hojas = []
    if "archivo_nombre" not in st.session_state:
        st.session_state.archivo_nombre = None

# ═════════════════════════════════════════════════════════════════════════════
# COMPONENTES DE UI
# ═════════════════════════════════════════════════════════════════════════════

def mostrar_logo_sidebar():
    """Muestra logo y encabezado en la barra lateral"""
    
    # Buscar logo
    posibles_rutas = [
        Path("assets/logocolegio.png"),
        Path("logocolegio.png"),
        Path("logo.png"),
    ]
    
    logo_encontrado = False
    for ruta_logo in posibles_rutas:
        if ruta_logo.exists():
            try:
                st.sidebar.image(str(ruta_logo), width=150)
                logo_encontrado = True
                break
            except:
                continue
    
    if not logo_encontrado:
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 15px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px; margin-bottom: 10px;'>
            <div style='font-size: 60px; margin: 10px 0;'>🏫</div>
            <p style='color: white; margin: 0; font-size: 12px;'>I.E. Víctor Núñez Valencia</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 20px;'>
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 30px; border-radius: 15px;'>
            <h2 style='margin: 0;'>🎓</h2>
            <h3 style='margin: 5px 0; font-size: 14px;'>Sistema Académico</h3>
            <h3 style='margin: 0; font-size: 14px;'>MINEDU 2025</h3>
        </div>
        <p style='color: #666; font-size: 11px; margin: 5px 0;'>{INFO_INSTITUCION.get('version', 'v5.1')}</p>
        <p style='color: #999; font-size: 10px; margin: 0;'>Powered by Alan Turing 🧠</p>
    </div>
    """, unsafe_allow_html=True)

def mostrar_cargador_archivos():
    """Muestra el cargador de archivos en la sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("📂 Cargar Datos")
    
    archivo_subido = st.sidebar.file_uploader(
        "Seleccionar archivo Excel",
        type=["xlsx", "xls"],
        help="Archivo con calificaciones de estudiantes (Nivel de logro - I Bimestre)",
        key="file_uploader"
    )
    
    if archivo_subido is not None:
        if st.session_state.archivo_nombre != archivo_subido.name:
            with st.sidebar:
                with st.spinner("📊 Procesando archivo..."):
                    datos_cargados, datos_raw, nombres_hojas, error = cargar_excel(archivo_subido)
                    
                    if error:
                        st.error(f"❌ {error}")
                        st.session_state.datos_cargados = None
                        st.session_state.datos_raw = None
                        st.session_state.nombres_hojas = []
                    else:
                        st.session_state.datos_cargados = datos_cargados
                        st.session_state.datos_raw = datos_raw
                        st.session_state.nombres_hojas = nombres_hojas
                        st.session_state.archivo_nombre = archivo_subido.name
                        st.success(f"✅ Cargado: {len(nombres_hojas)} hoja(s)")

def mostrar_menu_navegacion():
    """Muestra el menú de navegación en la sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("🧭 Navegación")
    
    # Definir páginas disponibles
    paginas = {
        "🏠 Inicio": "inicio",
        "👨‍🏫 Vista Director": "director",
        "👩‍🏫 Vista Docente": "docente",
        "🧑‍🎓 Análisis Estudiantil": "estudiantil",
        "🎯 Análisis Priorizados": "priorizados",
        "🤖 Modelo Predictivo ML": "predictivo",
        "📄 Exportar Reportes": "reportes",
        "❓ Ayuda": "ayuda"
    }
    
    seleccion = st.sidebar.radio(
        "Seleccione una vista:",
        options=list(paginas.keys()),
        label_visibility="collapsed"
    )
    
    return paginas[seleccion]

def mostrar_info_sidebar():
    """Muestra información adicional en la sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 10px; background: #f0f2f6; border-radius: 10px;'>
        <small>
            📚 Sistema basado en<br/>
            <b>PEI MINEDU 2024-2027</b><br/>
            {INFO_INSTITUCION.get('version', 'v5.1')}
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    # Estado de módulos
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📦 Módulos Activos:**")
    
    try:
        import sklearn
        st.sidebar.markdown("✅ scikit-learn (ML)")
    except:
        st.sidebar.markdown("❌ scikit-learn")
    
    try:
        import plotly
        st.sidebar.markdown("✅ Plotly (Gráficos)")
    except:
        st.sidebar.markdown("❌ Plotly")

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINAS
# ═════════════════════════════════════════════════════════════════════════════

def pagina_inicio():
    """Página de inicio del sistema"""
    st.title("🎓 Sistema de Análisis Académico V5.1")
    st.caption("✅ Arquitectura Modular Profesional - Optimizado por Alan Turing 🧠")
    
    st.markdown(f"""
    ### Bienvenido al Sistema Integrado MINEDU
    
    **Institución Educativa:**
    - 🏫 **{INFO_INSTITUCION.get('nombre_ie1', 'I.E.')}**
    - 📍 **{INFO_INSTITUCION.get('ubicacion', '')}**
    - 🔢 **Código:** {INFO_INSTITUCION.get('codigo', '')}
    - 📚 **{INFO_INSTITUCION.get('nivel', '')}**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📋 Guía Rápida:**
        1. Cargue archivo Excel (.xlsx/.xls)
        2. Sistema detecta hojas automáticamente
        3. Navegue por las vistas disponibles
        4. Exporte reportes cuando necesite
        
        **✨ Arquitectura Modular:**
        - Código organizado y mantenible
        - Fácil de extender y personalizar
        - Optimizado para rendimiento
        """)
    
    with col2:
        st.success("""
        **✨ Funcionalidades Activas:**
        ✅ Vista Director (Análisis Global)
        ✅ Vista Docente (Por Aula)
        ✅ Métricas ML (ROC, F1-Score)
        ✅ Análisis de Priorizados
        ✅ Tabla de Frecuencias
        ✅ Mapas de Calor
        ✅ Gráficos Interactivos
        ✅ Exportación Profesional
        
        **📊 Mejoras V5.1:**
        - Arquitectura modular SOLID
        - Métricas de Machine Learning
        - Clasificación de desaprobados
        - Análisis por áreas curriculares
        """)
    
    # Características técnicas
    st.markdown("---")
    st.markdown("### 🏗️ Arquitectura del Sistema")
    
    col_arq1, col_arq2, col_arq3 = st.columns(3)
    
    with col_arq1:
        st.markdown("""
        **📁 Módulos del Sistema:**
        - `constantes.py` - Config global
        - `utils.py` - Funciones auxiliares
        - `contexto.py` - Gestión de estado
        - `procesamiento.py` - Datos
        - `analisis_ml.py` - ML
        - `visualizaciones.py` - Gráficos
        """)
    
    with col_arq2:
        st.markdown("""
        **🎯 Vistas Principales:**
        - `vista_director.py` - Global
        - `vista_docente.py` - Por aula
        - `paginas_auxiliares.py` - Extras
        - `app_dashboard.py` - Principal
        """)
    
    with col_arq3:
        st.markdown("""
        **🛠️ Tecnologías:**
        - Python 3.8+
        - Streamlit
        - Pandas & NumPy
        - Plotly
        - scikit-learn
        - openpyxl
        """)
    
    st.markdown("---")
    
    if not st.session_state.datos_cargados:
        st.warning("""
        👈 **Para comenzar:** Use el botón **"Browse files"** en la barra lateral 
        para cargar su archivo Excel con las calificaciones.
        
        📝 **Formatos soportados:** .xlsx, .xls
        """)
    else:
        st.success("✅ Datos cargados correctamente. Use el menú de navegación para explorar las diferentes vistas.")

# ═════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """
    Función principal de la aplicación
    Punto de entrada del sistema
    """
    
    # Inicializar estado
    inicializar_session_state()
    
    # Mostrar sidebar
    mostrar_logo_sidebar()
    mostrar_cargador_archivos()
    
    # Obtener selección de página
    pagina_seleccionada = mostrar_menu_navegacion()
    
    # Mostrar información adicional
    mostrar_info_sidebar()
    
    # Renderizar página seleccionada
    try:
        if pagina_seleccionada == "inicio":
            pagina_inicio()
            
        elif pagina_seleccionada == "director":
            if st.session_state.datos_cargados:
                pagina_vista_director(
                    st.session_state.datos_cargados,
                    st.session_state.datos_raw
                )
            else:
                st.warning("⚠️ Por favor, cargue un archivo Excel primero.")
                pagina_inicio()
        
        elif pagina_seleccionada == "docente":
            if st.session_state.datos_cargados:
                pagina_vista_docente(st.session_state.datos_cargados)
            else:
                st.warning("⚠️ Por favor, cargue un archivo Excel primero.")
                pagina_inicio()
        
        elif pagina_seleccionada == "estudiantil":
            if st.session_state.datos_cargados:
                pagina_analisis_estudiantil(st.session_state.datos_cargados)
            else:
                st.warning("⚠️ Por favor, cargue un archivo Excel primero.")
                pagina_inicio()
        
        elif pagina_seleccionada == "priorizados":
            if st.session_state.datos_raw:
                pagina_analisis_priorizados(st.session_state.datos_raw)
            else:
                st.warning("⚠️ Por favor, cargue un archivo Excel primero.")
                pagina_inicio()
        
        elif pagina_seleccionada == "predictivo":
            if st.session_state.datos_raw:
                pagina_modelo_predictivo(st.session_state.datos_raw)
            else:
                st.warning("⚠️ Por favor, cargue un archivo Excel primero.")
                pagina_inicio()
        
        elif pagina_seleccionada == "reportes":
            if st.session_state.datos_cargados:
                pagina_exportar_reportes(st.session_state.datos_cargados)
            else:
                st.warning("⚠️ Por favor, cargue un archivo Excel primero.")
                pagina_inicio()
        
        elif pagina_seleccionada == "ayuda":
            pagina_ayuda()
        
        else:
            pagina_inicio()
    
    except Exception as e:
        st.error(f"❌ Error al cargar la página: {str(e)}")
        with st.expander("🔍 Ver detalles del error"):
            import traceback
            st.code(traceback.format_exc())

# ═════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Error crítico en la aplicación: {str(e)}")
        st.markdown("""
        ### 🔧 Solución de Problemas
        
        1. Verifique que todas las dependencias estén instaladas
        2. Asegúrese de que todos los archivos del proyecto estén presentes
        3. Revise los logs de error arriba
        4. Si el problema persiste, contacte al soporte técnico
        
        **Email:** ievinvasecundaria@gmail.com
        """)
        
        with st.expander("🔍 Información técnica del error"):
            import traceback
            st.code(traceback.format_exc())
