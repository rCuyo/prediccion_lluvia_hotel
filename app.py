from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Cargar modelos al iniciar
try:
    modelo = joblib.load('modelo_estado_fecha.pkl')
    le = joblib.load('label_encoder_estado.pkl')
    print(" Modelo y LabelEncoder cargados correctamente.")
except Exception as e:
    print(f" Error cargando modelos: {e}")
    modelo = None
    le = None

def predecir_estado(fecha, modelo, le):
    """Predice el estado del clima para una fecha dada"""
    fecha_dt = pd.to_datetime(fecha)
    
    X_new = pd.DataFrame({
        'year': [fecha_dt.year],
        'month': [fecha_dt.month],
        'day': [fecha_dt.day],
        'dayofyear': [fecha_dt.day_of_year],
        'weekofyear': [fecha_dt.isocalendar().week],
        'season': [("Verano" if fecha_dt.month in [12,1,2]
                    else "Otoño" if fecha_dt.month in [3,4,5]
                    else "Invierno" if fecha_dt.month in [6,7,8]
                    else "Primavera")]
    })
    
    pred = modelo.predict(X_new)
    estado = le.inverse_transform(pred)[0]
    
    return estado

def obtener_info_estado(estado):
    """Retorna información adicional según el estado"""
    info = {
        "Bueno": {
            "color": "#10b981",
            "icono": "☀️",
            "descripcion": "Excelente para turismo",
            "recomendacion": "Día perfecto para actividades al aire libre"
        },
        "Regular": {
            "color": "#f59e0b",
            "icono": "⛅",
            "descripcion": "Condiciones aceptables",
            "recomendacion": "Llevar protección adicional recomendada"
        },
        "Malo": {
            "color": "#ef4444",
            "icono": "🌧️",
            "descripcion": "No recomendado para turismo",
            "recomendacion": "Considere reprogramar actividades al aire libre"
        }
    }
    return info.get(estado, info["Regular"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        data = request.get_json()
        fecha_inicio = data.get('fecha_inicio')
        fecha_fin = data.get('fecha_fin')
        
        if not fecha_inicio or not fecha_fin:
            return jsonify({'error': 'Fechas no proporcionadas'}), 400
        
        if modelo is None or le is None:
            return jsonify({'error': 'Modelos no cargados correctamente'}), 500
        
        # Convertir fechas
        inicio = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fin = datetime.strptime(fecha_fin, '%Y-%m-%d')
        
        # Validar que la fecha fin sea mayor a la fecha inicio
        if fin < inicio:
            return jsonify({'error': 'La fecha final debe ser posterior a la inicial'}), 400
        
        # Limitar a 30 días máximo
        dias_diferencia = (fin - inicio).days
        if dias_diferencia > 30:
            return jsonify({'error': 'El rango máximo es de 30 días'}), 400
        
        # Generar predicciones para cada día
        predicciones = []
        fecha_actual = inicio
        
        while fecha_actual <= fin:
            estado = predecir_estado(fecha_actual.strftime('%Y-%m-%d'), modelo, le)
            info = obtener_info_estado(estado)
            
            predicciones.append({
                'fecha': fecha_actual.strftime('%Y-%m-%d'),
                'dia': fecha_actual.strftime('%d'),
                'mes': fecha_actual.strftime('%b'),
                'diaSemana': fecha_actual.strftime('%A'),
                'estado': estado,
                'color': info['color'],
                'icono': info['icono'],
                'descripcion': info['descripcion'],
                'recomendacion': info['recomendacion']
            })
            
            fecha_actual += timedelta(days=1)
        
        # Calcular estadísticas
        total_dias = len(predicciones)
        buenos = sum(1 for p in predicciones if p['estado'] == 'Bueno')
        regulares = sum(1 for p in predicciones if p['estado'] == 'Regular')
        malos = sum(1 for p in predicciones if p['estado'] == 'Malo')
        
        return jsonify({
            'predicciones': predicciones,
            'estadisticas': {
                'total': total_dias,
                'buenos': buenos,
                'regulares': regulares,
                'malos': malos,
                'porcentaje_bueno': round((buenos/total_dias)*100, 1)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)