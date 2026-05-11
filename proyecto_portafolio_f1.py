import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.ensemble import RandomForestRegressor
from streamlit_option_menu import option_menu

# ==========================================
# 1. CARGA DE DATOS Y MODELO
# ==========================================
@st.cache_data
def cargar_datos():
    ruta = 'tiendaventas.csv'
    df = pd.read_csv(ruta)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Año'] = df['Fecha'].dt.year
    df['Mes'] = df['Fecha'].dt.month
    df['Dia_Num'] = df['Fecha'].dt.dayofweek
    return df

df = cargar_datos()

@st.cache_resource
def entrenar_modelo(datos):
    x = datos[['Mes', 'Dia_Num', 'Festivo', 'Promociones']]
    y = datos['Ventas']
    modelo = RandomForestRegressor(random_state=42, n_estimators=100)
    modelo.fit(x, y)
    return modelo

modelo_ml = entrenar_modelo(df)

# Diccionario global para usar en varias pestañas
meses_nombres = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, 
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8, 
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
meses_abrev = {1:"Ene", 2:"Feb", 3:"Mar", 4:"Abr", 5:"May", 6:"Jun", 7:"Jul", 8:"Ago", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dic"}

# ==========================================
# 2. CONFIGURACIÓN E INYECCIÓN CSS
# ==========================================
st.set_page_config(page_title="App de Ventas & ML", layout="wide")

st.markdown("""
    <style>
    /* 1. Títulos de las tarjetas */
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] span,
    div[data-testid="stMetricLabel"] div {
        font-size: 24px !important;  
        font-weight: bold !important;
    }
    
    /* 2. Valor principal de la tarjeta */
    div[data-testid="stMetricValue"] > div {
        font-size: 45px !important;  
    }
    
    /* 3. Porcentaje y texto inferior */
    div[data-testid="stMetricDelta"] > div {
        font-size: 18px !important;  
    }

    /* 4. NUEVO: Títulos de los filtros y selectores (Ej: "Selecciona el rango...") */
    div[data-testid="stWidgetLabel"] p {
        font-size: 25px !important;  /* <- Cambia este número al tamaño que prefieras */
        font-weight: bold !important; /* Opcional: quítalo si no lo quieres en negrita */
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MENÚ LATERAL MODERNO
# ==========================================
with st.sidebar:
    st.title("Pestañas")
    
    opcion_menu = option_menu(
        menu_title=None, 
        options=["Dashboard Histórico", "Predicción a 7 Días", "Simulador de Estrategia", "Detección de Anomalías"],
        icons=["bar-chart-fill", "magic", "sliders", "exclamation-triangle-fill"], 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4A628A", "color": "white", "font-weight": "bold"},
        }
    )
    
    st.markdown("---")
    if st.button("🛑 Finalizar Sesión"):
        st.success("Servidor apagado. Puedes cerrar la pestaña.")
        os._exit(0)

st.title("🤖 Plataforma de Análisis y Predicción de Ventas")

# ==========================================
# PESTAÑA 1: DASHBOARD HISTÓRICO
# ==========================================
if opcion_menu == "Dashboard Histórico":
    st.header("Análisis de Ventas")
    
    col_fecha1, col_fecha2 = st.columns([1, 3])
    with col_fecha1:
        st.markdown("### Selecciona el rango de fechas:")
        rango_fechas = st.date_input(
            label="",
            value=(df['Fecha'].min().date(), df['Fecha'].max().date()),
            min_value=df['Fecha'].min().date(),
            max_value=df['Fecha'].max().date()
        )
    
    if len(rango_fechas) == 2:
        fecha_inicio, fecha_fin = rango_fechas
        df_filtrado = df[(df['Fecha'].dt.date >= fecha_inicio) & (df['Fecha'].dt.date <= fecha_fin)]
        
        dias_seleccionados = (fecha_fin - fecha_inicio).days + 1
        ventas_actuales = df_filtrado['Ventas'].sum()
        
        # Lógica de comparación de tarjeta
        if dias_seleccionados >= 360:
            ventas_enero = df[df['Mes'] == 1]['Ventas'].sum()
            ventas_dic = df[df['Mes'] == 12]['Ventas'].sum()
            if ventas_enero > 0:
                crec_anual = ((ventas_dic - ventas_enero) / ventas_enero) * 100
                delta_texto = f"{crec_anual:.1f}% Ene vs Dic"
            else:
                delta_texto = "0%"
        else:
            # Compara el último mes del rango seleccionado vs el mes anterior
            mes_fin = fecha_fin.month
            año_fin = fecha_fin.year
            
            if mes_fin == 1:
                mes_ant = 12
                año_ant = año_fin - 1
            else:
                mes_ant = mes_fin - 1
                año_ant = año_fin
                
            ventas_ultimo_mes = df[(df['Año'] == año_fin) & (df['Mes'] == mes_fin)]['Ventas'].sum()
            ventas_mes_anterior = df[(df['Año'] == año_ant) & (df['Mes'] == mes_ant)]['Ventas'].sum()
            
            if ventas_mes_anterior > 0:
                crecimiento = ((ventas_ultimo_mes - ventas_mes_anterior) / ventas_mes_anterior) * 100
                delta_texto = f"{crecimiento:.1f}% {meses_abrev[mes_fin]} vs {meses_abrev[mes_ant]} "
            else:
                delta_texto = "Sin datos mes anterior"

        # Mostrar tarjetas
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            # Determinamos el color del porcentaje (Verde si es positivo, Rojo si es negativo)
            # Buscamos si el primer caracter de delta_texto es un número o un '-'
            es_positivo = not delta_texto.startswith('-') and "0%" not in delta_texto
            color_delta = '#28a745' if es_positivo else '#dc3545'
            simbolo = '▲' if es_positivo else '▼'

            st.markdown(f"""
                <div style="line-height: 1.2; background-color: #1e2130; padding: 20px; border-radius: 10px;">
                    <p style="font-size: 30px; font-weight: bold; margin-bottom: 5px; color: white;">Ventas Totales</p>
                    <p style="font-size: 45px; margin-top: 0px; margin-bottom: 5px; color: white;">${ventas_actuales:,.0f}</p>
                    <p style="font-size: 18px; color: {color_delta}; font-weight: bold;">{simbolo} {delta_texto}</p>
                </div>
            """, unsafe_allow_html=True)

        with col_metric2:
            st.markdown(f"""
                <div style="line-height: 1.2; background-color: #1e2130; padding: 20px; border-radius: 10px;">
                    <p style="font-size: 30px; font-weight: bold; margin-bottom: 5px; color: white;">Total Días Festivos</p>
                    <p style="font-size: 45px; margin-top: 0px; margin-bottom: 5px; color: white;">{df_filtrado['Festivo'].sum()}</p>
                    <p style="font-size: 18px; color: #888;">Días calendario</p>
                </div>
            """, unsafe_allow_html=True)

        with col_metric3:
            st.markdown(f"""
                <div style="line-height: 1.2; background-color: #1e2130; padding: 20px; border-radius: 10px;">
                    <p style="font-size: 30px; font-weight: bold; margin-bottom: 5px; color: white;">Total Días Promoción</p>
                    <p style="font-size: 45px; margin-top: 0px; margin-bottom: 5px; color: white;">{df_filtrado['Promociones'].sum()}</p>
                    <p style="font-size: 18px; color: #888;">Días campaña</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        ventas_mes = df_filtrado.groupby('Mes')['Ventas'].sum().reset_index()
        fig_mes = px.line(ventas_mes, x='Mes', y='Ventas', markers=True, title='Ventas Totales por Mes (Log)',
                          log_y=True, text='Ventas')
        # ---> AÑADE ESTA LÍNEA PARA EL TÍTULO DEL GRÁFICO <---
        fig_mes.update_layout(title_font_size=24)
        fig_mes.update_traces(texttemplate='%{text:.3f}', textposition='top center')
        st.plotly_chart(fig_mes, use_container_width=True) 
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        ventas_dia = df_filtrado.groupby('DíaDeLaSemana')['Ventas'].mean().reset_index()
        fig_dia = px.bar(ventas_dia, x='DíaDeLaSemana', y='Ventas', title='Promedio de Ventas por Día (Log)',
                         log_y=True, text_auto='.3f')
        fig_dia.update_layout(title_font_size=24)
        fig_dia.update_traces(textposition='outside')
        st.plotly_chart(fig_dia, use_container_width=True)
    else:
        st.warning("Por favor, selecciona una fecha de inicio y una fecha de fin.")

# ==========================================
# PESTAÑA 2: PREDICCIÓN (FORECASTING)
# ==========================================
elif opcion_menu == "Predicción a 7 Días":
    st.header("Predicción de Ventas Semanales")
    st.write("Selecciona a partir de qué fecha quieres simular la semana de ventas.")
    
    col_pred1, col_pred2 = st.columns([1, 3])
    with col_pred1:
        st.markdown("#### Fecha de inicio de la semana:")
        fecha_inicio_pred = st.date_input("", value=df['Fecha'].max().date() + pd.Timedelta(days=1))
    
    # Generar los 7 días a partir de la fecha elegida
    fechas_futuras = pd.date_range(start=fecha_inicio_pred, periods=7)
    
    df_futuro = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Mes': fechas_futuras.month,
        'Dia_Num': fechas_futuras.dayofweek,
        'Festivo': 0,      
        'Promociones': 0   
    })
    
    df_futuro['Ventas'] = modelo_ml.predict(df_futuro[['Mes', 'Dia_Num', 'Festivo', 'Promociones']])
    df_futuro['Tipo'] = 'Predicción'
    
    # Extraer los 14 días reales justo anteriores a esa fecha para comparar visualmente
    df_reciente = df[df['Fecha'] < pd.to_datetime(fecha_inicio_pred)].tail(14).copy()
    df_reciente['Tipo'] = 'Real'
    
    df_combinado = pd.concat([df_reciente, df_futuro])
    
    fig_pred = px.line(df_combinado, x='Fecha', y='Ventas', color='Tipo', markers=True, 
                       line_dash='Tipo', title='Histórico vs Semana Seleccionada (Log)',
                       log_y=True, text='Ventas')
    fig_pred.update_layout(title_font_size=24)
    fig_pred.update_traces(texttemplate='%{text:.3f}', textposition='top right')
    st.plotly_chart(fig_pred, use_container_width=True)

# ==========================================
# PESTAÑA 3: SIMULADOR DE ESTRATEGIA
# ==========================================
elif opcion_menu == "Simulador de Estrategia":
    st.header("Simulador de Escenarios")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("#### Selecciona el Mes:")
        seleccion_mes = st.selectbox("", list(meses_nombres.keys()))
        sim_mes = meses_nombres[seleccion_mes]
        
        nombres_dias = {
            "Lunes": 0, "Martes": 1, "Miércoles": 2, 
            "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6
        }
        st.markdown("#### Día de la semana:")
        seleccion_dia = st.selectbox("", list(nombres_dias.keys()))
        sim_dia = nombres_dias[seleccion_dia] 
        
        sim_promo = st.checkbox("¿Activar Promoción de Marketing?")
        sim_festivo = st.checkbox("¿Es día Festivo?")
    
    promo_val = 1 if sim_promo else 0
    fest_val = 1 if sim_festivo else 0
    
    datos_usuario = pd.DataFrame([[sim_mes, sim_dia, fest_val, promo_val]], 
                                 columns=['Mes', 'Dia_Num', 'Festivo', 'Promociones'])
    
    prediccion_usuario = modelo_ml.predict(datos_usuario)[0]
    
    with col_sim2:
        st.info("Resultado de la Predicción")
        st.metric(label="Ventas Estimadas", value=f"${prediccion_usuario:,.3f}")

# ==========================================
# PESTAÑA 4: DETECTOR DE ANOMALÍAS
# ==========================================
elif opcion_menu == "Detección de Anomalías":
    st.header("Detector de Anomalías")
    
    df['Prediccion_Base'] = modelo_ml.predict(df[['Mes', 'Dia_Num', 'Festivo', 'Promociones']])
    df['Error'] = abs(df['Ventas'] - df['Prediccion_Base'])
    df['Es_Anomalia'] = np.where(df['Error'] > 1800, 'Anomalía', 'Normal')
    
    fig_anomalia = px.scatter(df, x='Fecha', y='Ventas', color='Es_Anomalia', 
                              color_discrete_map={'Normal': '#1f77b4', 'Anomalía': 'red'},
                              title="Días con Comportamiento Anormal de Ventas (Log)",
                              log_y=True, labels={'Es_Anomalia': 'Tipo de Día'})
    fig_anomalia.update_layout(title_font_size=24,
                            legend_title_font_size=20,      # Tamaño del título de la leyenda (ej. "Es_Anomalia")
                            legend_font_size=18      )       # Tamaño de los ítems de la leyenda (ej. "Normal"))
    st.plotly_chart(fig_anomalia, use_container_width=True)