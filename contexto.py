# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                      CONTEXTO Y CONFIGURACIÓN GLOBAL                      ║
║             Gestión centralizada de configuraciones del sistema           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

from constantes import (
    ESCALA_CALIFICACIONES,
    ESTRATEGIAS_MINEDU,
    INFO_INSTITUCION,
    EQUIVALENCIAS_NOTAS
)

class GestorEvaluacionMINEDU:
    """
    Gestiona conversiones y análisis de calificaciones según normativa MINEDU
    """
    
    def __init__(self):
        self.escala = ESCALA_CALIFICACIONES
        self.estrategias = ESTRATEGIAS_MINEDU
        self._cache_conversiones = {}
    
    def num_a_letra(self, valor: float) -> str:
        """Convierte nota numérica (0-20) a letra (C/B/A/AD)"""
        import pandas as pd
        
        if pd.isna(valor):
            return "C"
        
        valor_redondeado = round(float(valor), 2)
        
        if valor_redondeado in self._cache_conversiones:
            return self._cache_conversiones[valor_redondeado]
        
        for letra, config in self.escala.items():
            if config['min'] <= valor_redondeado <= config['max']:
                self._cache_conversiones[valor_redondeado] = letra
                return letra
        
        return "C"
    
    def letra_a_num(self, letra: str) -> float:
        """Convierte letra (C/B/A/AD) a valor numérico representativo"""
        letra = str(letra).strip().upper()
        return float(self.escala.get(letra, {'num': 8})['num'])
    
    def get_color(self, letra: str) -> str:
        """Retorna color hexadecimal para la letra"""
        return self.escala.get(letra, {}).get('color', '#999999')
    
    def generar_observacion(self, promedio: float, nombre: str = "el estudiante") -> tuple:
        """Genera observación pedagógica completa"""
        letra = self.num_a_letra(promedio)
        config = self.escala[letra]
        estrategia = self.estrategias[letra]
        
        observacion = f"""
**👤 Estudiante:** {nombre}
**📊 Promedio:** {promedio:.2f}/20.00
**📈 Nivel:** {letra} - {config['desc']}

**📋 Observación Pedagógica:**
{estrategia}
"""
        return observacion, letra

# Instancia global del gestor
gestor_evaluacion = GestorEvaluacionMINEDU()
