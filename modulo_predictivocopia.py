
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║            MÓDULO PREDICTIVO - MACHINE LEARNING CON CATBOOST             ║
║                         Autor: frederickv                                ║
║                        Fecha: 2025-11-11 (Corregido v2)                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import yaml
import re
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier, Pool
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✅ modulo_predictivo.py cargado exitosamente")

# ═════════════════════════════════════════════════════════════════════════════
# CLASE 1: GESTOR DE EVALUACIÓN MINEDU
# ═════════════════════════════════════════════════════════════════════════════

class GestorEvaluacionMINEDU:
    """Gestiona la conversión y evaluación de calificaciones según MINEDU"""
    
    PROFILES_YAML = """
version: 2025-11-08
profiles:
  MINEDU_NUEVO_2024:
    letters_order: [AD, A, B, C]
    numeric_scale: [0, 20]
    levels:
      AD:
        range: [18, 20]
        representative: 19
        name: "Logro Destacado"
        desc: "Demuestra aprendizajes que superan el nivel esperado"
      A:
        range: [15, 17]
        representative: 16
        name: "Logro Esperado"
        desc: "Demuestra el nivel esperado respecto a la competencia"
      B:
        range: [11, 14]
        representative: 12
        name: "En Proceso"
        desc: "Está próximo a alcanzar el nivel esperado"
      C:
        range: [0, 10]
        representative: 8
        name: "En Inicio"
        desc: "Muestra un progreso mínimo en la competencia"
    pass_letters: [AD, A, B]
    fail_letters: [C]
"""
    
    ESTRATEGIAS = {
        "C": ("🚨 **Reforzamiento Urgente Requerido**\n"
              "• Implementar Plan de Tutoría Individualizado (PTI)\n"
              "• Foco en competencias básicas con sesiones de 30-45 min\n"
              "• Contactar a padres/apoderados para acompañamiento familiar\n"),
        "B": ("⚠️ **Acompañamiento Pedagógico Necesario**\n"
              "• Proporcionar material didáctico diferenciado\n"
              "• Fomentar trabajo colaborativo (grupos de 3-4 estudiantes)\n"),
        "A": ("✅ **Consolidación de Aprendizaje**\n"
              "• Asignar proyectos de aplicación práctica (ABP)\n"
              "• Promover resolución de problemas complejos\n"),
        "AD": ("🌟 **Potenciación de Talento Excepcional**\n"
               "• Fomentar proyectos de investigación autónomos\n"
               "• Asignar rol de tutor par (mentoría entre estudiantes)\n")
    }

    def __init__(self, profile_name: str = "MINEDU_NUEVO_2024"):
        self.profiles = yaml.safe_load(self.PROFILES_YAML)
        self.profile_name = profile_name
        self.cfg = self.profiles["profiles"][profile_name]
        self.levels = self.cfg["levels"]
        self.letters_order = self.cfg["letters_order"]
        self.pass_letters = set(self.cfg["pass_letters"])
        self.fail_letters = set(self.cfg.get("fail_letters", []))
        
        # Propiedades adicionales para compatibilidad
        self.letras_aprobadas = self.pass_letters
        self.letras_reprobadas = self.fail_letters
        
        self._cache_num_to_letter = {}
        logger.info(f"✓ Gestor de Evaluación MINEDU inicializado (Versión: {self.profiles['version']})")

    def letra_a_num(self, letter: str) -> float:
        """Convierte letra a valor numérico"""
        letter = str(letter).strip().upper()
        if letter in self.levels:
            return float(self.levels[letter]["representative"])
        return float(self.levels["C"]["representative"])
    
    def convertir_letra_a_num(self, letter: str) -> float:
        """Alias para compatibilidad"""
        return self.letra_a_num(letter)

    def num_a_letra(self, value: float) -> str:
        """Convierte valor numérico a letra"""
        if pd.isna(value):
            return "C"
        value = round(float(value), 1)
        if value in self._cache_num_to_letter:
            return self._cache_num_to_letter[value]
        for letra in self.letters_order:
            lo, hi = self.levels[letra]["range"]
            if lo <= value <= hi:
                self._cache_num_to_letter[value] = letra
                return letra
        return "C"
    
    def convertir_num_a_letra(self, value: float) -> str:
        """Alias para compatibilidad"""
        return self.num_a_letra(value)

# ═════════════════════════════════════════════════════════════════════════════
# CLASE 2: LIMPIADOR DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

class LimpiarRankingEstudiantes_v3:
    """Limpia y prepara datos crudos de Excel para análisis"""
    
    def __init__(self, df_raw: pd.DataFrame, hoja: str, perfil: GestorEvaluacionMINEDU):
        self.df_raw = df_raw
        self.hoja = hoja
        self.perfil = perfil
        self.letter_set = set(perfil.letters_order)
        self.bad_colnames = self.letter_set.union({"TOTAL", "4", "3", "2", "1"})
        self.df = None
        logger.info("✓ Limpiador v3 inicializado")

    def _encontrar_fila_encabezado(self) -> int:
        """Busca la fila que contiene el encabezado"""
        pat = re.compile(r"APELLIDOS.*ESTUDIANTE|NOMBRES.*APELLIDOS", re.I)
        for i in range(min(50, len(self.df_raw))):
            # 🔧 CORRECCIÓN: Aplicar .str solo a Series, no a DataFrame
            row_data = self.df_raw.iloc[i].astype(str)
            row_strs = row_data.str.replace("\n", " ")
            
            if row_strs.str.contains(pat, na=False).any():
                logger.info(f"✓ Encabezado de estudiantes encontrado en fila {i}")
                return i
        
        logger.warning("No se encontró encabezado por nombre, buscando por contenido...")
        
        # Método alternativo: buscar fila con más letras de calificación
        def contar_letras_validas(row):
            return sum(1 for x in row if str(x).strip().upper() in self.letter_set)
        
        counts = self.df_raw.apply(contar_letras_validas, axis=1)
        
        if counts.max() > 0:
            fila_datos = counts.idxmax()
            fila_encabezado = max(0, fila_datos - 1)
            logger.info(f"✓ Encabezado por contenido detectado en fila {fila_encabezado}")
            return fila_encabezado
            
        raise ValueError("No se pudo detectar la fila del encabezado en la hoja")

    def _limpiar_columnas(self):
        """Elimina columnas vacías o innecesarias"""
        cols_mantener = []
        for c in self.df.columns:
            c_str = str(c).strip().lower()
            if c_str != 'nan' and not c_str.startswith('unnamed') and len(c_str) > 0:
                cols_mantener.append(c)
        
        if len(cols_mantener) == 0:
            logger.warning("⚠️ No se encontraron columnas válidas, manteniendo todas")
            return
            
        self.df = self.df[cols_mantener].copy()
        self.df.columns = (
            self.df.columns.astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def _procesar_columna_estudiante(self) -> List[str]:
        """Identifica y procesa la columna de nombres de estudiantes"""
        id_cols = []
        name_col = None
        for c in self.df.columns:
            if re.search(r"APELLIDOS|NOMBRES|ESTUDIANTE", str(c), re.I):
                name_col = c
                break
        
        if name_col:
            self.df["ESTUDIANTE"] = self.df[name_col].astype(str).str.strip()
            id_cols.append("ESTUDIANTE")
            logger.info(f"✓ Columna ESTUDIANTE creada desde: {name_col}")
        else:
            # Si no se encuentra, usar la primera columna
            first_col = self.df.columns[0]
            self.df["ESTUDIANTE"] = self.df[first_col].astype(str).str.strip()
            id_cols.append("ESTUDIANTE")
            logger.warning(f"⚠️ Usando primera columna como ESTUDIANTE: {first_col}")
            
        for col in ["GRADO", "SECCION", "AULA", "NIVEL"]:
            if col in self.df.columns:
                id_cols.append(col)
        return id_cols

    def _detectar_columnas_calificaciones(self, id_cols: List[str]) -> List[str]:
        """Detecta columnas que contienen calificaciones"""
        candidate_cols = []
        for c in self.df.columns:
            if str(c).strip().upper() in self.bad_colnames or c in id_cols:
                continue
            
            vals = self.df[c].dropna()
            if len(vals) < 0.3 * len(self.df):  # Reducido umbral a 30%
                continue
            
            # 🔧 CORRECCIÓN CRÍTICA: Convertir correctamente a string
            try:
                vals_str = vals.apply(lambda x: str(x).strip().upper())
                pct_letters = vals_str.isin(self.letter_set).mean()
                
                if pct_letters >= 0.5:  # Reducido umbral a 50%
                    candidate_cols.append(c)
            except Exception as e:
                logger.warning(f"⚠️ Error procesando columna {c}: {e}")
                continue
                
        logger.info(f"✓ {len(candidate_cols)} columnas de calificaciones detectadas")
        if len(candidate_cols) > 0:
            logger.info(f"   Ejemplos: {candidate_cols[:5]}")
        return candidate_cols

    def procesar(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Procesa el DataFrame completo y retorna datos limpios"""
        logger.info(f"🚀 Iniciando limpieza de hoja '{self.hoja}'...")
        
        try:
            header_row = self._encontrar_fila_encabezado()
        except ValueError as e:
            logger.error(f"❌ {str(e)}")
            # Intentar con la primera fila como header
            header_row = 0
            logger.warning("⚠️ Usando fila 0 como encabezado por defecto")
        
        # 🔧 CORRECCIÓN: Aplicar .str solo a Series
        header_series = self.df_raw.iloc[header_row].astype(str)
        header = header_series.str.replace("\n", " ").str.strip()
        
        self.df = self.df_raw.iloc[header_row + 1:].copy()
        self.df.columns = header
        
        self._limpiar_columnas()
        cols_id = self._procesar_columna_estudiante()
        
        # Filtrar filas vacías
        non_id_cols = [c for c in self.df.columns if c not in cols_id]
        if len(non_id_cols) > 0:
            self.df = self.df.dropna(how="all", subset=non_id_cols)
        
        self.df = self.df.dropna(subset=['ESTUDIANTE']).reset_index(drop=True)
        
        # Filtrar estudiantes vacíos o inválidos
        self.df = self.df[self.df['ESTUDIANTE'].str.strip() != ''].reset_index(drop=True)
        
        cols_calificaciones = self._detectar_columnas_calificaciones(cols_id)
        
        # Normalizar valores solo en columnas de calificación
        for col in cols_calificaciones:
            # 🔧 CORRECCIÓN: Usar apply en lugar de .str directamente
            self.df[col] = self.df[col].apply(lambda x: str(x).strip().upper())
            # Reemplazar valores no válidos con 'C' (reprobado por defecto)
            self.df[col] = self.df[col].apply(
                lambda x: x if x in self.letter_set else 'C'
            )
            
        logger.info(f"✅ Limpieza completada: {len(self.df)} estudiantes, {len(cols_calificaciones)} calificaciones")
        return self.df, cols_calificaciones, cols_id

# (El resto del código permanece igual...)
# Copio las clases restantes sin cambios:

class ModeloAprobacionEstudiantes:
    """Modelo de Machine Learning para predecir aprobación de estudiantes"""
    
    def __init__(self, columnas_calificaciones: List[str], perfil: GestorEvaluacionMINEDU):
        self.columnas_calificaciones = columnas_calificaciones
        self.perfil = perfil
        self.model = None
        self.cat_feature_indices = list(range(len(columnas_calificaciones)))
        logger.info("✓ Modelo de Aprobación inicializado")

    def _calcular_aprobacion(self, row: pd.Series) -> int:
        """Calcula si un estudiante aprueba según sus calificaciones"""
        vals = row[self.columnas_calificaciones].values
        reprobados = sum(1 for v in vals if v in self.perfil.letras_reprobadas)
        return int(reprobados == 0)

    def preparar_datos(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara los datos para entrenamiento"""
        logger.info("📊 Preparando datos para el modelo...")
        X = df[self.columnas_calificaciones].astype(str).copy()
        y = df.apply(self._calcular_aprobacion, axis=1).astype(int)
        
        balance = y.value_counts(normalize=True)
        logger.info(f"Distribución del target: Aprobados={balance.get(1, 0):.1%}, Reprobados={balance.get(0, 0):.1%}")
        
        if len(y.unique()) < 2:
            logger.warning("⚠️ Solo hay una clase en el target. El modelo no se puede entrenar.")
            
        return X, y

    def entrenar_modelo(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Entrena el modelo CatBoost"""
        logger.info("🚀 Entrenando modelo CatBoost...")
        
        if len(y.unique()) < 2:
            logger.error("❌ No se puede entrenar: solo hay una clase en los datos")
            return {"error": "Datos insuficientes para entrenamiento"}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        train_pool = Pool(X_train, y_train, cat_features=self.cat_feature_indices)
        test_pool = Pool(X_test, y_test, cat_features=self.cat_feature_indices)
        
        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            auto_class_weights='Balanced'
        )
        
        self.model.fit(train_pool, eval_set=test_pool, verbose=False)
        
        logger.info("✅ Entrenamiento completado.")
        
        pred_prob = self.model.predict_proba(test_pool)[:, 1]
        pred_cls = (pred_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, pred_prob)
        
        logger.info(f"📈 RESULTADOS DEL MODELO:")
        logger.info(f"   🎯 AUC-ROC: {auc:.4f}")
        
        reporte = classification_report(
            y_test, pred_cls, 
            target_names=["DESAPROBADO", "APROBADO"],
            zero_division=0
        )
        logger.info("\n" + reporte)
        
        return {
            "model": self.model,
            "auc": auc,
            "X_test": X_test,
            "y_test": y_test,
            "reporte_clasificacion": classification_report(
                y_test, pred_cls, output_dict=True, zero_division=0
            )
        }

class GeneradorReporteEstudiantes:
    """Genera reportes detallados con predicciones y observaciones"""
    
    def __init__(self, model, columnas_calificaciones: List[str], 
                 cols_id: List[str], perfil: GestorEvaluacionMINEDU):
        self.model = model
        self.columnas_calificaciones = columnas_calificaciones
        self.cols_id = cols_id
        self.perfil = perfil
        self.cat_feature_indices = list(range(len(columnas_calificaciones)))
        logger.info("✓ Generador de Reportes inicializado")

    def _construir_observacion(self, row: pd.Series) -> str:
        """Construye observación pedagógica basada en el nivel"""
        promedio_num = self.perfil.convertir_letra_a_num(row['CALIFICACION_ML'])
        observacion_base = self.perfil.ESTRATEGIAS.get(row['CALIFICACION_ML'], "Sin observación")
        return f"Nivel: {row['CALIFICACION_ML']} ({promedio_num:.1f}) - {observacion_base}"

    def generar_reporte_completo(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Genera el reporte completo con predicciones"""
        logger.info("📋 Generando reporte final de estudiantes...")
        
        all_pool = Pool(X, cat_features=self.cat_feature_indices)
        all_prob = self.model.predict_proba(all_pool)[:, 1]
        all_pred = (all_prob >= 0.5).astype(int)
        
        reporte = df[self.cols_id].copy()
        reporte['PRED_APROBADO'] = all_pred
        reporte['PROB_APROBADO'] = (all_prob * 100).round(1)
        reporte['ESTADO_PREDICHO'] = reporte['PRED_APROBADO'].map({
            1: "APROBADO", 
            0: "EN RIESGO (DESAPROBADO)"
        })
        
        def calc_prom_num(row):
            valores = [self.perfil.letra_a_num(row[c]) for c in self.columnas_calificaciones]
            return np.mean(valores)
            
        reporte['PROMEDIO_NUM_ML'] = X.apply(calc_prom_num, axis=1)
        reporte['CALIFICACION_ML'] = reporte['PROMEDIO_NUM_ML'].apply(
            self.perfil.convertir_num_a_letra
        )
        
        reporte['OBSERVACION_PEDAGOGICA'] = reporte.apply(
            self._construir_observacion, axis=1
        )
        
        reporte_final = reporte.sort_values(by="PROB_APROBADO", ascending=True)
        
        logger.info("✅ Reporte final generado.")
        return reporte_final

class AnalizadorImportanciaFeatures:
    """Analiza y visualiza la importancia de características del modelo"""
    
    def __init__(self, model, columnas_features: List[str]):
        if not hasattr(model, 'is_fitted') or not model.is_fitted():
            raise ValueError("El modelo no está entrenado.")
        self.model = model
        self.columnas_features = columnas_features
        self.df_importance = None
        logger.info("✓ Analizador de Importancia inicializado")

    def obtener_importancia(self) -> pd.DataFrame:
        """Obtiene la importancia de cada característica"""
        importances = self.model.get_feature_importance()
        self.df_importance = pd.DataFrame({
            'feature': self.columnas_features,
            'importance': importances,
            'importance_pct': (importances / importances.sum() * 100).round(2)
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        return self.df_importance

    def visualizar_importancia(self, top_n: int = 15, show_plot: bool = False) -> plt.Figure:
        """Genera visualización de importancia de características"""
        if self.df_importance is None:
            self.obtener_importancia()
        
        logger.info(f"📊 Generando visualización de importancia (Top {top_n})...")
        
        data = self.df_importance.head(top_n).iloc[::-1]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
        
        bars = ax.barh(range(len(data)), data['importance'], color=colors, edgecolor='black')
        
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['feature'], fontsize=10)
        ax.set_xlabel('Importancia (PredictionValuesChange)', fontsize=12)
        ax.set_title(f'Top {top_n} Características más Importantes', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, pct in zip(bars, data['importance_pct']):
            ax.text(bar.get_width() + 0.01 * data['importance'].max(), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', 
                    va='center', 
                    fontsize=9, 
                    fontweight='bold')
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
            
        return fig

def ejecutar_analisis_predictivo(df_raw_ie: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """Ejecuta el pipeline completo de ML en la hoja 'IE'"""
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO ANÁLISIS PREDICTIVO CON MACHINE LEARNING")
    logger.info("=" * 80)
    
    try:
        perfil = GestorEvaluacionMINEDU()
        limpiador = LimpiarRankingEstudiantes_v3(df_raw_ie, "IE", perfil)
        df_limpio, cols_cal, cols_id = limpiador.procesar()
        
        if len(cols_cal) == 0:
            raise ValueError("No se detectaron columnas de calificaciones válidas")
        
        if len(df_limpio) < 10:
            raise ValueError(f"Datos insuficientes: solo {len(df_limpio)} estudiantes encontrados")
        
        modelo = ModeloAprobacionEstudiantes(cols_cal, perfil)
        X, y = modelo.preparar_datos(df_limpio)
        resultados_entrenamiento = modelo.entrenar_modelo(X, y)
        
        if "error" in resultados_entrenamiento:
            raise ValueError(resultados_entrenamiento["error"])
        
        generador_reporte = GeneradorReporteEstudiantes(
            modelo.model, 
            cols_cal, 
            cols_id, 
            perfil
        )
        df_reporte = generador_reporte.generar_reporte_completo(df_limpio, X)
        
        analizador_importancia = AnalizadorImportanciaFeatures(
            modelo.model, 
            cols_cal
        )
        fig_importancia = analizador_importancia.visualizar_importancia(
            top_n=min(15, len(cols_cal)), 
            show_plot=False
        )
        
        logger.info("=" * 80)
        logger.info("✅ PIPELINE DE ANÁLISIS PREDICTIVO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        
        return df_reporte, fig_importancia
    
    except Exception as e:
        logger.error(f"❌ Error durante el análisis predictivo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    print("=" * 80)
    print("MÓDULO PREDICTIVO - MACHINE LEARNING CON CATBOOST")
    print("=" * 80)
    print("\n✅ Módulo cargado correctamente")