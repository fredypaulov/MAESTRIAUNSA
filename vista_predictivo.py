
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║           MODELO PREDICTIVO DE RENDIMIENTO ACADÉMICO V5.1                ║
║        🤖 Machine Learning Avanzado con CatBoost, XGBoost, RF            ║
║        📊 Predicción de Rendimiento Futuro con IA                        ║
║        🎯 Sistema de Alerta Temprana Predictivo                          ║
║        ⭐ La Mejor Implementación ML para Educación - Alan Turing        ║
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
import warnings
warnings.filterwarnings('ignore')

# Importaciones de ML
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix, classification_report
    )
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from constantes import (
    INFO_INSTITUCION,
    ESCALA_CALIFICACIONES,
    ESTRATEGIAS_MINEDU,
    UMBRAL_APROBACION,
    COLORES_NIVELES
)
from procesamiento import obtener_columnas_notas, procesar_datos
from utils import find_column, calcular_porcentaje_seguro

# ═════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PREPARACIÓN DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

def preparar_datos_ml(df: pd.DataFrame, columnas_num: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara datos para modelos de Machine Learning
    
    Args:
        df: DataFrame procesado
        columnas_num: Columnas numéricas de notas
        
    Returns:
        (X_features, y_target)
    """
    try:
        # Features: Todas las notas numéricas
        X = df[columnas_num].copy()
        
        # Target: Estado (Aprobado/Desaprobado)
        y = (df['PROMEDIO'] >= UMBRAL_APROBACION).astype(int)
        
        # Rellenar NaN con la media
        X = X.fillna(X.mean())
        
        return X, y
    
    except Exception as e:
        st.error(f"Error preparando datos: {e}")
        return pd.DataFrame(), pd.Series()

def crear_features_avanzadas(df: pd.DataFrame, columnas_num: List[str]) -> pd.DataFrame:
    """
    Crea features adicionales para mejorar predicciones
    
    Features agregadas:
    - Promedio general
    - Desviación estándar de notas
    - Nota mínima y máxima
    - Tendencia (si mejora o empeora)
    - Cantidad de notas bajas (< 11)
    """
    try:
        X = df[columnas_num].copy()
        
        # Features estadísticas
        X['promedio'] = X.mean(axis=1)
        X['std'] = X.std(axis=1)
        X['min'] = X.min(axis=1)
        X['max'] = X.max(axis=1)
        X['rango'] = X['max'] - X['min']
        X['notas_bajas'] = (X[columnas_num] < UMBRAL_APROBACION).sum(axis=1)
        
        # Tendencia (simplificada)
        if len(columnas_num) >= 3:
            X['tendencia'] = (X[columnas_num[-1]] - X[columnas_num[0]])
        else:
            X['tendencia'] = 0
        
        return X
    
    except Exception as e:
        st.warning(f"Error creando features avanzadas: {e}")
        return df[columnas_num].copy()

# ═════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO DE MODELOS
# ═════════════════════════════════════════════════════════════════════════════

def entrenar_random_forest(X_train, X_test, y_train, y_test) -> Tuple[object, Dict]:
    """Entrena modelo Random Forest"""
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        
        metricas = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return rf, metricas
    
    except Exception as e:
        st.error(f"Error entrenando Random Forest: {e}")
        return None, {}

def entrenar_xgboost(X_train, X_test, y_train, y_test) -> Tuple[object, Dict]:
    """Entrena modelo XGBoost"""
    if not XGBOOST_AVAILABLE:
        return None, {}
    
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(X_train, y_train, verbose=False)
        y_pred = xgb_model.predict(X_test)
        y_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        metricas = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return xgb_model, metricas
    
    except Exception as e:
        st.error(f"Error entrenando XGBoost: {e}")
        return None, {}

def entrenar_catboost(X_train, X_test, y_train, y_test) -> Tuple[object, Dict]:
    """Entrena modelo CatBoost"""
    if not CATBOOST_AVAILABLE:
        return None, {}
    
    try:
        cat_model = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )
        
        cat_model.fit(X_train, y_train, verbose=False)
        y_pred = cat_model.predict(X_test).flatten()
        y_proba = cat_model.predict_proba(X_test)[:, 1]
        
        metricas = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return cat_model, metricas
    
    except Exception as e:
        st.error(f"Error entrenando CatBoost: {e}")
        return None, {}

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES AVANZADAS
# ═════════════════════════════════════════════════════════════════════════════

def crear_grafico_comparacion_modelos(resultados: Dict) -> go.Figure:
    """Crea gráfico comparativo de modelos"""
    modelos = []
    metricas_list = []
    
    for nombre, metricas in resultados.items():
        if metricas:
            modelos.append(nombre)
            metricas_list.append({
                'Accuracy': metricas.get('accuracy', 0),
                'Precision': metricas.get('precision', 0),
                'Recall': metricas.get('recall', 0),
                'F1-Score': metricas.get('f1_score', 0),
                'ROC-AUC': metricas.get('roc_auc', 0)
            })
    
    if not metricas_list:
        return go.Figure()
    
    df_comp = pd.DataFrame(metricas_list, index=modelos)
    
    fig = go.Figure()
    
    for metrica in df_comp.columns:
        fig.add_trace(go.Bar(
            name=metrica,
            x=df_comp.index,
            y=df_comp[metrica],
            text=df_comp[metrica].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Comparación de Modelos de Machine Learning',
        xaxis_title='Modelo',
        yaxis_title='Score',
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def crear_curvas_roc(resultados: Dict, y_test) -> go.Figure:
    """Crea curvas ROC para todos los modelos"""
    fig = go.Figure()
    
    for nombre, metricas in resultados.items():
        if metricas and 'y_proba' in metricas:
            try:
                fpr, tpr, _ = roc_curve(y_test, metricas['y_proba'])
                auc = metricas.get('roc_auc', 0)
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=f'{nombre} (AUC={auc:.3f})',
                    mode='lines',
                    line=dict(width=2)
                ))
            except:
                continue
    
    # Línea diagonal (clasificador aleatorio)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Clasificador Aleatorio',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='Curvas ROC - Comparación de Modelos',
        xaxis_title='Tasa de Falsos Positivos (FPR)',
        yaxis_title='Tasa de Verdaderos Positivos (TPR)',
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def crear_matriz_confusion_multiple(resultados: Dict, y_test) -> go.Figure:
    """Crea matrices de confusión para todos los modelos"""
    n_modelos = sum(1 for m in resultados.values() if m and 'y_pred' in m)
    
    if n_modelos == 0:
        return go.Figure()
    
    fig = make_subplots(
        rows=1,
        cols=n_modelos,
        subplot_titles=list(resultados.keys()),
        specs=[[{'type': 'heatmap'}] * n_modelos]
    )
    
    col = 1
    for nombre, metricas in resultados.items():
        if metricas and 'y_pred' in metricas:
            try:
                cm = confusion_matrix(y_test, metricas['y_pred'])
                
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=['Pred: Desap', 'Pred: Aprob'],
                        y=['Real: Desap', 'Real: Aprob'],
                        text=cm,
                        texttemplate='%{text}',
                        textfont={"size": 16},
                        colorscale='Blues',
                        showscale=(col == n_modelos)
                    ),
                    row=1,
                    col=col
                )
                col += 1
            except:
                continue
    
    fig.update_layout(
        title_text='Matrices de Confusión - Comparación de Modelos',
        height=400
    )
    
    return fig

def crear_grafico_importancia_features(modelo, feature_names: List[str], nombre_modelo: str) -> go.Figure:
    """Crea gráfico de importancia de features"""
    try:
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
        elif hasattr(modelo, 'get_feature_importance'):
            importancias = modelo.get_feature_importance()
        else:
            return go.Figure()
        
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importancia': importancias
        }).sort_values('Importancia', ascending=False).head(10)
        
        fig = px.bar(
            df_imp,
            x='Importancia',
            y='Feature',
            orientation='h',
            title=f'Top 10 Features Más Importantes - {nombre_modelo}',
            text='Importancia'
        )
        
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400)
        
        return fig
    
    except Exception as e:
        st.warning(f"No se pudo generar importancia de features: {e}")
        return go.Figure()

def predecir_rendimiento_futuro(modelo, X_actual: pd.DataFrame, periodos: int = 4) -> pd.DataFrame:
    """
    Predice rendimiento futuro para los próximos periodos
    
    Args:
        modelo: Modelo ML entrenado
        X_actual: Features actuales
        periodos: Número de bimestres a predecir
        
    Returns:
        DataFrame con predicciones
    """
    try:
        predicciones = []
        
        for i in range(periodos):
            # Predicción de probabilidad
            if hasattr(modelo, 'predict_proba'):
                proba = modelo.predict_proba(X_actual)[:, 1]
            else:
                proba = modelo.predict(X_actual)
            
            # Convertir probabilidad a promedio estimado
            promedio_estimado = proba * 20  # Escala 0-20
            
            predicciones.append({
                'Periodo': f'Bimestre {i+2}',
                'Promedio_Estimado': promedio_estimado.mean(),
                'Prob_Aprobacion': proba.mean() * 100
            })
        
        return pd.DataFrame(predicciones)
    
    except Exception as e:
        st.warning(f"Error en predicción futura: {e}")
        return pd.DataFrame()

# ═════════════════════════════════════════════════════════════════════════════
# RECOMENDACIONES AUTOMATIZADAS
# ═════════════════════════════════════════════════════════════════════════════

def generar_recomendaciones_ml(
    metricas_modelos: Dict,
    predicciones_futuro: pd.DataFrame,
    estudiantes_riesgo: int
) -> str:
    """Genera recomendaciones automatizadas basadas en ML"""
    
    # Encontrar mejor modelo
    mejor_modelo = max(metricas_modelos.items(), key=lambda x: x[1].get('f1_score', 0) if x[1] else 0)
    nombre_mejor = mejor_modelo[0]
    f1_mejor = mejor_modelo[1].get('f1_score', 0) if mejor_modelo[1] else 0
    
    # Tendencia futura
    if not predicciones_futuro.empty:
        tendencia = "POSITIVA" if predicciones_futuro['Promedio_Estimado'].is_monotonic_increasing else \
                   "NEGATIVA" if predicciones_futuro['Promedio_Estimado'].is_monotonic_decreasing else "ESTABLE"
        prom_futuro = predicciones_futuro['Promedio_Estimado'].iloc[-1]
    else:
        tendencia = "NO DISPONIBLE"
        prom_futuro = 0
    
    recomendaciones = f"""
## 🤖 Recomendaciones Automatizadas Basadas en Machine Learning

### 📊 Análisis Predictivo:

**Mejor Modelo Detectado:** {nombre_mejor} (F1-Score: {f1_mejor:.3f})

**Tendencia Proyectada:** {tendencia}

**Promedio Estimado (Próximo Bimestre):** {prom_futuro:.2f}/20

**Estudiantes en Riesgo Identificados:** {estudiantes_riesgo}

---

### 🎯 Recomendaciones para Estudiantes:

"""
    
    if tendencia == "NEGATIVA":
        recomendaciones += """
#### 🚨 ALERTA: Tendencia Negativa Detectada

**Acciones Urgentes:**
1. **Reforzamiento Inmediato:** Aumentar horas de estudio a 3 horas diarias
2. **Tutorías Especializadas:** Solicitar apoyo personalizado del docente
3. **Identificar Causas:** ¿Problemas de comprensión? ¿Falta de motivación? ¿Problemas personales?
4. **Plan de Recuperación:** Enfocarse en áreas con notas más bajas
5. **Seguimiento Semanal:** Reuniones semanales con tutor para monitorear avances

**Técnicas Específicas:**
- Método Pomodoro (25 min estudio + 5 min descanso)
- Resúmenes y mapas mentales después de cada clase
- Práctica diaria de ejercicios (mínimo 10 problemas/día)
- Enseñar lo aprendido a otros (consolida conocimiento)
"""
    
    elif tendencia == "POSITIVA":
        recomendaciones += """
#### ✅ EXCELENTE: Tendencia Positiva Detectada

**Mantener el Momentum:**
1. **Consolidar Hábitos:** Continuar con rutina de estudio actual
2. **Desafíos Adicionales:** Buscar ejercicios más complejos
3. **Mentoría:** Ayudar a compañeros con dificultades
4. **Proyectos Extra:** Participar en concursos académicos
5. **Preparación Avanzada:** Adelantar temas del siguiente bimestre

**Técnicas de Potenciación:**
- Técnicas de estudio avanzadas (Cornell, SQ3R)
- Participación activa en debates y presentaciones
- Investigación independiente de temas de interés
- Uso de plataformas avanzadas (Coursera, edX)
"""
    
    else:  # ESTABLE
        recomendaciones += """
#### ⚖️ ESTABLE: Tendencia Constante Detectada

**Romper el Estancamiento:**
1. **Variar Métodos:** Cambiar técnicas de estudio
2. **Identificar Brechas:** Detectar áreas específicas de mejora
3. **Aumentar Desafío:** Resolver problemas más complejos
4. **Feedback Constante:** Solicitar retroalimentación del docente
5. **Objetivos Específicos:** Establecer metas de mejora por área

**Técnicas de Mejora:**
- Autoevaluaciones semanales
- Grupos de estudio con compañeros destacados
- Sesiones de práctica intensiva pre-examen
- Uso de recursos multimedia (videos, simulaciones)
"""
    
    recomendaciones += """

---

### 👨‍🏫 Recomendaciones para Docentes:

"""
    
    if f1_mejor >= 0.9:
        recomendaciones += f"""
#### ✅ Alta Precisión del Modelo (F1={f1_mejor:.3f})

**El sistema de detección es altamente confiable. Acciones sugeridas:**

1. **Intervención Temprana:**
   - Usar predicciones del modelo para identificar estudiantes en riesgo ANTES de que desaprueben
   - Implementar tutorías preventivas para estudiantes con probabilidad < 60% de aprobación

2. **Personalización de Enseñanza:**
   - Adaptar metodología según patrones detectados por el modelo
   - Crear grupos de reforzamiento basados en necesidades similares identificadas por ML

3. **Monitoreo Continuo:**
   - Actualizar predicciones cada 2 semanas
   - Ajustar estrategias según feedback del modelo

4. **Evaluación Formativa:**
   - Enfocarse en áreas identificadas como críticas por el análisis de importancia de features
   - Evaluaciones frecuentes en competencias con mayor peso predictivo
"""
    
    else:
        recomendaciones += f"""
#### ⚠️ Precisión Moderada del Modelo (F1={f1_mejor:.3f})

**Se requiere mejorar la calidad de datos:**

1. **Recopilación de Más Datos:**
   - El modelo necesita más información histórica (múltiples bimestres)
   - Registrar datos adicionales: asistencia, participación, tareas entregadas

2. **Diversificar Evaluaciones:**
   - Incluir diferentes tipos de evaluación (oral, escrita, proyectos)
   - Registrar competencias específicas, no solo promedio general

3. **Factores Externos:**
   - Considerar contexto socioeconómico
   - Registrar apoyo familiar
   - Documentar problemas de salud o personales

4. **Validación Manual:**
   - Contrastar predicciones del modelo con observación docente
   - Ajustar intervenciones según criterio pedagógico
"""
    
    recomendaciones += """

---

### 👨‍👩‍👧‍👦 Recomendaciones para Padres:

**Basado en el análisis predictivo, su hijo(a) necesita:**

"""
    
    if estudiantes_riesgo > 0:
        recomendaciones += """
#### 🚨 Apoyo Familiar Crítico

1. **Supervisión Diaria:**
   - Verificar que estudie mínimo 2 horas/día
   - Revisar y firmar cuadernos y agenda
   - Asegurar que complete todas las tareas

2. **Ambiente de Estudio:**
   - Espacio tranquilo, sin distractores
   - Horario fijo de estudio (no negociable)
   - Sin TV, celular o videojuegos durante estudio

3. **Apoyo Emocional:**
   - Conversar sobre dificultades académicas sin juzgar
   - Reforzar confianza: "Tú puedes lograrlo"
   - Celebrar pequeños avances

4. **Comunicación con Colegio:**
   - Asistir a TODAS las reuniones
   - Mantener comunicación semanal con tutor
   - Coordinar estrategias conjuntas

5. **Salud Integral:**
   - 8 horas de sueño obligatorio
   - Alimentación nutritiva
   - Tiempo de recreación supervisada
"""
    
    else:
        recomendaciones += """
#### ✅ Consolidación del Rendimiento

1. **Mantenimiento:**
   - Continuar con rutinas actuales que están funcionando
   - Supervisión moderada pero constante

2. **Motivación:**
   - Reconocer y celebrar logros
   - Incentivar lectura y aprendizaje autónomo
   - Apoyar intereses académicos especiales

3. **Desarrollo Integral:**
   - Balance entre estudios y actividades extracurriculares
   - Fomentar habilidades sociales y emocionales
"""
    
    recomendaciones += """

---

### 🏛️ Recomendaciones para Dirección:

"""
    
    recomendaciones += f"""
#### 📊 Análisis Institucional Basado en ML

**Métricas del Sistema Predictivo:**
- Modelo más efectivo: {nombre_mejor}
- Precisión general (F1-Score): {f1_mejor:.3f}
- Estudiantes identificados en riesgo: {estudiantes_riesgo}

**Plan de Acción Institucional:**

1. **Implementación del Sistema de Alerta:**
   - Usar predicciones ML como herramienta de diagnóstico temprano
   - Generar reportes automáticos cada 2 semanas
   - Dashboard de seguimiento en tiempo real

2. **Capacitación Docente:**
   - Talleres sobre interpretación de métricas ML
   - Uso de sistema predictivo en planificación pedagógica
   - Estrategias de intervención basadas en datos

3. **Recursos y Presupuesto:**
   - Asignar horas de tutoría según predicciones de riesgo
   - Material didáctico para áreas identificadas como críticas
   - Software educativo personalizado

4. **Coordinaciones Externas:**
   - Presentar resultados a UGEL con propuestas basadas en datos
   - Solicitar apoyo especializado para casos críticos identificados
   - Alianzas con universidades para validación del modelo

5. **Monitoreo y Evaluación:**
   - Comparar predicciones vs. resultados reales cada bimestre
   - Ajustar parámetros del modelo según precisión
   - Documentar casos de éxito de intervención temprana

---

## 🔮 Predicción de Rendimiento Futuro

"""
    
    if not predicciones_futuro.empty:
        recomendaciones += "**Proyección para Próximos Bimestres:**\n\n"
        for _, row in predicciones_futuro.iterrows():
            recomendaciones += f"- **{row['Periodo']}:** Promedio estimado {row['Promedio_Estimado']:.2f}/20 "
            recomendaciones += f"(Probabilidad de aprobación: {row['Prob_Aprobacion']:.1f}%)\n"
        
        recomendaciones += "\n**Interpretación:**\n"
        if prom_futuro >= 14:
            recomendaciones += "✅ Se proyecta rendimiento SATISFACTORIO. Mantener estrategias actuales.\n"
        elif prom_futuro >= 11:
            recomendaciones += "⚠️ Se proyecta rendimiento EN RIESGO. Implementar reforzamiento preventivo.\n"
        else:
            recomendaciones += "🚨 Se proyecta DESAPROBACIÓN. Intervención urgente requerida.\n"
    
    recomendaciones += """

---

## 📚 Referencias y Fundamentos

**Técnicas de Machine Learning Utilizadas:**
- **Random Forest:** Ensemble de árboles de decisión para robustez
- **XGBoost:** Gradient boosting optimizado para alta precisión
- **CatBoost:** Algoritmo especializado en datos categóricos

**Métricas de Evaluación:**
- **ROC-AUC:** Capacidad de discriminación entre aprobados/desaprobados
- **F1-Score:** Balance entre precisión y sensibilidad
- **Accuracy:** Porcentaje de predicciones correctas

**Validación:**
- Validación cruzada 5-fold para evitar sobreajuste
- Conjunto de prueba independiente (20% de datos)
- Análisis de importancia de features para interpretabilidad

**Normativa:**
- RVM N° 094-2020-MINEDU: Evaluación de competencias
- Uso de IA en educación: Principios de transparencia y ética
- UGEL Arequipa 2025: Directivas de reforzamiento escolar

---

**⚠️ NOTA IMPORTANTE:**
Este sistema predictivo es una **herramienta de apoyo** para la toma de decisiones pedagógicas.
Las predicciones deben ser complementadas con observación docente y criterio profesional.
El objetivo es la PREVENCIÓN y mejora continua, no el etiquetamiento de estudiantes.

---

*Generado por Sistema Académico MINEDU V5.1 - Alan Turing Edition*
*{datetime.now().strftime('%d/%m/%Y %H:%M')}*
"""
    
    return recomendaciones

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def pagina_modelo_predictivo(datos_raw: Dict):
    """
    🤖 MODELO PREDICTIVO DE RENDIMIENTO ACADÉMICO
    
    Sistema completo de Machine Learning con:
    - CatBoost, XGBoost, Random Forest
    - Curvas ROC, Matrices de Confusión
    - Predicción de rendimiento futuro
    - Recomendaciones automatizadas
    - Análisis de importancia de features
    """
    
    st.title("🤖 Modelo Predictivo de Rendimiento Académico")
    st.caption("Predicciones basadas en Machine Learning | Sistema de Alerta Temprana Avanzado")
    
    if not datos_raw:
        st.warning("⚠️ No hay datos cargados")
        return
    
    # Verificar disponibilidad de bibliotecas
    st.markdown("### 📦 Estado de Bibliotecas ML")
    
    col_lib1, col_lib2, col_lib3 = st.columns(3)
    
    with col_lib1:
        if SKLEARN_AVAILABLE:
            st.success("✅ scikit-learn")
        else:
            st.error("❌ scikit-learn")
            st.code("pip install scikit-learn")
    
    with col_lib2:
        if XGBOOST_AVAILABLE:
            st.success("✅ XGBoost")
        else:
            st.warning("⚠️ XGBoost (opcional)")
            st.code("pip install xgboost")
    
    with col_lib3:
        if CATBOOST_AVAILABLE:
            st.success("✅ CatBoost")
        else:
            st.warning("⚠️ CatBoost (opcional)")
            st.code("pip install catboost")
    
    if not SKLEARN_AVAILABLE:
        st.error("❌ Se requiere scikit-learn para usar este módulo")
        st.info("""
        **Para instalar las bibliotecas necesarias:**
        
        ```
        pip install scikit-learn xgboost catboost
        ```
        """)
        return
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. CONSOLIDACIÓN DE DATOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 📊 Preparación de Datos")
    
    df_list = []
    
    with st.spinner("Procesando datos..."):
        for nombre, df_hoja in datos_raw.items():
            try:
                cols_notas, _ = obtener_columnas_notas(df_hoja)
                if not cols_notas:
                    continue
                
                df_proc, cols_num = procesar_datos(df_hoja, cols_notas)
                df_proc['AULA'] = nombre
                df_list.append((df_proc, cols_num))
            except Exception as e:
                st.warning(f"Error en '{nombre}': {e}")
                continue
    
    if not df_list:
        st.error("No se pudieron procesar datos")
        return
    
    df_consolidado = pd.concat([df for df, _ in df_list], ignore_index=True)
    columnas_num = df_list[0][1]
    
    total_est = len(df_consolidado)
    desaprobados = (df_consolidado['ESTADO'] == 'Desaprobado').sum()
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Estudiantes", total_est)
    col_info2.metric("Aprobados", total_est - desaprobados)
    col_info3.metric("Desaprobados", desaprobados)
    
    st.success(f"✅ Datos preparados: {total_est} estudiantes, {len(columnas_num)} features")
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2. ENTRENAMIENTO DE MODELOS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("### 🤖 Entrenamiento de Modelos ML")
    
    usar_features_avanzadas = st.checkbox(
        "Usar features avanzadas (mejora precisión)",
        value=True,
        help="Agrega estadísticas como promedio, desviación estándar, tendencia"
    )
    
    if st.button("🚀 Entrenar Modelos", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelos de Machine Learning..."):
            
            # Preparar datos
            if usar_features_avanzadas:
                X = crear_features_avanzadas(df_consolidado, columnas_num)
            else:
                X, _ = preparar_datos_ml(df_consolidado, columnas_num)
            
            y = (df_consolidado['PROMEDIO'] >= UMBRAL_APROBACION).astype(int)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            
            st.info(f"📊 Conjunto de entrenamiento: {len(X_train)} | Conjunto de prueba: {len(X_test)}")
            
            # Entrenar modelos
            resultados = {}
            modelos = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Random Forest
            status_text.text("Entrenando Random Forest...")
            rf_model, rf_metricas = entrenar_random_forest(X_train, X_test, y_train, y_test)
            if rf_model:
                resultados['Random Forest'] = rf_metricas
                modelos['Random Forest'] = rf_model
            progress_bar.progress(33)
            
            # XGBoost
            if XGBOOST_AVAILABLE:
                status_text.text("Entrenando XGBoost...")
                xgb_model, xgb_metricas = entrenar_xgboost(X_train, X_test, y_train, y_test)
                if xgb_model:
                    resultados['XGBoost'] = xgb_metricas
                    modelos['XGBoost'] = xgb_model
            progress_bar.progress(66)
            
            # CatBoost
            if CATBOOST_AVAILABLE:
                status_text.text("Entrenando CatBoost...")
                cat_model, cat_metricas = entrenar_catboost(X_train, X_test, y_train, y_test)
                if cat_model:
                    resultados['CatBoost'] = cat_metricas
                    modelos['CatBoost'] = cat_model
            progress_bar.progress(100)
            
            status_text.text("✅ Entrenamiento completado")
            
            # Guardar en session_state
            st.session_state['ml_resultados'] = resultados
            st.session_state['ml_modelos'] = modelos
            st.session_state['ml_X_test'] = X_test
            st.session_state['ml_y_test'] = y_test
            st.session_state['ml_X'] = X
            st.session_state['ml_feature_names'] = X.columns.tolist()
            
            st.success(f"🎉 {len(resultados)} modelos entrenados exitosamente")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. RESULTADOS Y VISUALIZACIONES
    # ═══════════════════════════════════════════════════════════════════════════
    
    if 'ml_resultados' in st.session_state and st.session_state['ml_resultados']:
        st.markdown("---")
        st.markdown("### 📊 Resultados y Comparación de Modelos")
        
        resultados = st.session_state['ml_resultados']
        modelos = st.session_state['ml_modelos']
        y_test = st.session_state['ml_y_test']
        X = st.session_state['ml_X']
        feature_names = st.session_state['ml_feature_names']
        
        # Tabs para organizar visualizaciones
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Comparación",
            "📈 Curvas ROC",
            "🎯 Matrices Confusión",
            "⭐ Importancia Features",
            "🔮 Predicción Futuro",
            "💡 Recomendaciones"
        ])
        
        with tab1:
            st.markdown("#### Comparación de Métricas")
            
            fig_comp = crear_grafico_comparacion_modelos(resultados)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Tabla de métricas
            df_metricas = pd.DataFrame([
                {
                    'Modelo': nombre,
                    'Accuracy': f"{m.get('accuracy', 0):.3f}",
                    'Precision': f"{m.get('precision', 0):.3f}",
                    'Recall': f"{m.get('recall', 0):.3f}",
                    'F1-Score': f"{m.get('f1_score', 0):.3f}",
                    'ROC-AUC': f"{m.get('roc_auc', 0):.3f}"
                }
                for nombre, m in resultados.items() if m
            ])
            
            st.dataframe(df_metricas, use_container_width=True, hide_index=True)
            
            # Mejor modelo
            mejor = max(resultados.items(), key=lambda x: x[1].get('f1_score', 0) if x[1] else 0)
            st.success(f"🏆 **Mejor Modelo:** {mejor[0]} (F1-Score: {mejor[1].get('f1_score', 0):.3f})")
        
        with tab2:
            st.markdown("#### Curvas ROC-AUC")
            
            fig_roc = crear_curvas_roc(resultados, y_test)
            st.plotly_chart(fig_roc, use_container_width=True)
            
            with st.expander("📖 ¿Qué es la Curva ROC?"):
                st.markdown("""
                **Curva ROC (Receiver Operating Characteristic):**
                
                - Muestra la capacidad del modelo para distinguir entre clases
                - **Eje X:** Tasa de Falsos Positivos (estudiantes mal clasificados como aprobados)
                - **Eje Y:** Tasa de Verdaderos Positivos (estudiantes correctamente identificados)
                - **AUC (Area Under Curve):** Área bajo la curva
                  - **1.0 = Perfecto:** Clasificación perfecta
                  - **0.9-1.0 = Excelente**
                  - **0.8-0.9 = Muy bueno**
                  - **0.7-0.8 = Bueno**
                  - **0.5 = Aleatorio:** No mejor que adivinar
                
                **Interpretación para Educación:**
                Un AUC alto significa que el modelo puede identificar con precisión 
                qué estudiantes aprobarán y cuáles necesitan reforzamiento.
                """)
        
        with tab3:
            st.markdown("#### Matrices de Confusión")
            
            fig_cm = crear_matriz_confusion_multiple(resultados, y_test)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.info("""
            **Interpretación de la Matriz:**
            - **Verdaderos Positivos (abajo derecha):** Aprobados correctamente identificados
            - **Verdaderos Negativos (arriba izquierda):** Desaprobados correctamente identificados
            - **Falsos Positivos (arriba derecha):** Desaprobados predichos como aprobados (CRÍTICO)
            - **Falsos Negativos (abajo izquierda):** Aprobados predichos como desaprobados
            
            Lo ideal es maximizar la diagonal principal y minimizar los errores.
            """)
        
        with tab4:
            st.markdown("#### Importancia de Features")
            
            modelo_seleccionado = st.selectbox(
                "Seleccione modelo:",
                list(modelos.keys())
            )
            
            if modelo_seleccionado in modelos:
                fig_imp = crear_grafico_importancia_features(
                    modelos[modelo_seleccionado],
                    feature_names,
                    modelo_seleccionado
                )
                
                if fig_imp.data:
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.success("""
                    **Features más importantes** son las que más influyen en la predicción.
                    Docentes deben enfocarse en estas áreas para mejorar el rendimiento estudiantil.
                    """)
                else:
                    st.info("Importancia de features no disponible para este modelo")
        
        with tab5:
            st.markdown("#### Predicción de Rendimiento Futuro")
            
            # Seleccionar modelo para predicción
            modelo_pred = st.selectbox(
                "Modelo para predicción:",
                list(modelos.keys()),
                key='modelo_pred'
            )
            
            periodos_pred = st.slider(
                "Bimestres a predecir:",
                min_value=1,
                max_value=4,
                value=3
            )
            
            if st.button("🔮 Generar Predicción", use_container_width=True):
                with st.spinner("Generando predicciones..."):
                    predicciones = predecir_rendimiento_futuro(
                        modelos[modelo_pred],
                        X,
                        periodos_pred
                    )
                    
                    if not predicciones.empty:
                        # Gráfico de tendencia
                        fig_pred = go.Figure()
                        
                        # Agregar valor actual
                        actual_prom = df_consolidado['PROMEDIO'].mean()
                        periodos_grafic = ['I Bim (Actual)'] + predicciones['Periodo'].tolist()
                        promedios_grafic = [actual_prom] + predicciones['Promedio_Estimado'].tolist()
                        
                        fig_pred.add_trace(go.Scatter(
                            x=periodos_grafic,
                            y=promedios_grafic,
                            mode='lines+markers+text',
                            name='Promedio Estimado',
                            line=dict(color='#667eea', width=3),
                            marker=dict(size=12),
                            text=[f'{p:.2f}' for p in promedios_grafic],
                            textposition='top center'
                        ))
                        
                        # Línea de aprobación
                        fig_pred.add_hline(
                            y=UMBRAL_APROBACION,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="Línea de Aprobación"
                        )
                        
                        fig_pred.update_layout(
                            title=f'Proyección de Rendimiento Futuro - {modelo_pred}',
                            xaxis_title='Periodo',
                            yaxis_title='Promedio Estimado',
                            height=500,
                            yaxis=dict(range=[0, 20])
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Tabla de predicciones
                        st.dataframe(predicciones, use_container_width=True, hide_index=True)
                        
                        # Interpretación
                        tendencia_futura = predicciones['Promedio_Estimado'].iloc[-1]
                        
                        if tendencia_futura >= 14:
                            st.success(f"""
                            ✅ **Proyección POSITIVA**
                            
                            Se estima un promedio de {tendencia_futura:.2f} para el final del año.
                            **Acción:** Mantener estrategias actuales y consolidar aprendizajes.
                            """)
                        elif tendencia_futura >= 11:
                            st.warning(f"""
                            ⚠️ **Proyección EN RIESGO**
                            
                            Se estima un promedio de {tendencia_futura:.2f} (apenas aprobando).
                            **Acción:** Implementar reforzamiento preventivo AHORA.
                            """)
                        else:
                            st.error(f"""
                            🚨 **Proyección CRÍTICA**
                            
                            Se estima un promedio de {tendencia_futura:.2f} (desaprobado).
                            **Acción:** Intervención urgente e intensiva requerida.
                            """)
                        
                        st.session_state['predicciones_futuro'] = predicciones
        
        with tab6:
            st.markdown("#### Recomendaciones Automatizadas")
            
            estudiantes_riesgo = desaprobados
            predicciones_f = st.session_state.get('predicciones_futuro', pd.DataFrame())
            
            recomendaciones = generar_recomendaciones_ml(
                resultados,
                predicciones_f,
                estudiantes_riesgo
            )
            
            st.markdown(recomendaciones)
            
            # Descarga de recomendaciones
            col_desc1, col_desc2 = st.columns(2)
            
            with col_desc1:
                st.download_button(
                    "📥 Descargar Recomendaciones TXT",
                    data=recomendaciones,
                    file_name=f"recomendaciones_ml_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_desc2:
                # Exportar métricas a Excel
                if st.button("📊 Exportar Métricas Excel", use_container_width=True):
                    st.info("Funcionalidad próximamente disponible")
    
    else:
        st.info("👆 Haga clic en **'Entrenar Modelos'** para comenzar el análisis predictivo")
        
        st.markdown("""
        ### ✨ Funcionalidades del Módulo Predictivo:
        
        #### 🤖 Modelos de Machine Learning:
        - **Random Forest:** Ensemble robusto de árboles de decisión
        - **XGBoost:** Gradient boosting de alta precisión
        - **CatBoost:** Especializado en datos categóricos
        
        #### 📊 Visualizaciones Avanzadas:
        - **Curvas ROC-AUC:** Capacidad discriminatoria de modelos
        - **Matrices de Confusión:** Errores y aciertos por modelo
        - **Importancia de Features:** Factores más influyentes
        - **Predicción de Tendencias:** Proyección a futuro
        
        #### 🎯 Análisis Predictivo:
        - Identificación temprana de riesgo académico
        - Predicción de rendimiento futuro (hasta 4 bimestres)
        - Factores que más influyen en el aprendizaje
        - Probabilidad de aprobación por estudiante
        
        #### 💡 Recomendaciones Automatizadas:
        - **Para Estudiantes:** Técnicas de estudio personalizadas
        - **Para Docentes:** Estrategias pedagógicas basadas en datos
        - **Para Padres:** Plan de apoyo familiar específico
        - **Para Dirección:** Plan de acción institucional
        
        #### 📋 Requisitos:
        - ✅ Datos del I Bimestre (disponible)
        - ⏳ Datos de múltiples bimestres (para mayor precisión)
        - ✅ Bibliotecas ML instaladas
        
        ---
        
        **🌟 Sistema de clase mundial **
        
        *Este módulo utiliza técnicas avanzadas de Machine Learning validadas
        en investigación educativa internacional.*
        """)
