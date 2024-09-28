# %%
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
import numpy as np
import base64
import io

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
from scipy.signal import find_peaks
import warnings
import webbrowser
warnings.filterwarnings("ignore")

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Inicializar la aplicación Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Layout de la aplicación
app.layout = html.Div([
    html.Img(src=app.get_asset_url('logo_adl.png'), height="200", width="200",),
    html.Img(src=app.get_asset_url('logo_fac.png'), height="200", width="200",),
    html.H1("Análisis de señales", style={'textAlign': 'center'}),
    html.H2("CODEFEST - ADASTRA", style={'textAlign': 'center'}),
    html.H2("ADL", style={'textAlign': 'center'}),
    html.H3("Cargue sus datos", style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data',
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        children=html.Button('Cargar CSV'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Br(),
    html.H3(id="output-senal", style={'textAlign': 'center'}),
    dcc.Graph(id='tabla2'),
    html.H3("Tabla de métricas", style={'textAlign': 'center'}),
    dcc.Graph(id='tabla'),
    dbc.Row([
            dbc.Col(
                [html.H3("Señales, ancho de banda y picos", style={'textAlign': 'center'}),
                 dcc.Graph(id='graph1'),
                 html.H3("Frecuencias espureas", style={'textAlign': 'center'}),
                    dcc.Graph(id='graph2')],
                    md=8),
            dbc.Col([
                html.H3("Espectrógrama de señales", style={'textAlign': 'center'}),
                dbc.Col(dcc.Graph(id='graph3'), md=4),
                ]),
    ]),
])

# Función para procesar el archivo CSV cargado
def parse_contents(contents, tipo):
    # Decodificar el contenido del CSV
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Leer el CSV en un DataFrame
    if tipo == 1:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';', skiprows=46, skipfooter=635, decimal=',')
        df.rename(columns={'Frequency [Hz]':'Frequency_Hz', 'Magnitude [dBm]':'Power_dBm'}, inplace=True)
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    if tipo == 2:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';', skiprows=679, header=None)
        df.drop(df[df.index >0].index, inplace=True)
    if tipo == 3:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';', skiprows=682, header=None, decimal=',')
    return df

# Callback para manejar la carga del archivo y actualizar los gráficos
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('graph3', 'figure'),
     Output('tabla', 'figure'),
     Output('tabla2', 'figure'),
     Output('output-senal', 'children'),
     ],
    [Input('upload-data', 'contents')]
)
def update_output(contents):
    if contents is None:
        return "No hay datos cargados", {}, {}, {}, {}, {}, {},
    # Procesar los datos
    df = parse_contents(contents, 1)
    df2 = parse_contents(contents, 2)
    df3 = parse_contents(contents, 3)
    df3.columns = df2.iloc[0,:]
    
    #CALCULO DE METRICAS
    analysis_df = pd.DataFrame(columns=['Frecuencia Central (MHz)', 'Frecuencia Central ponderada (MHz)', 'Ancho de banda (Hz)', 'Amplitud/Potencia (mW)', 'Nivel de ruido DBm', 'Relación señal ruido SNR (dB)',
                  'Frecuencias de espurea', 'Frecuencias armonicas (MHz)', 'Modulación', 'Picos espectrales', 'Ancho de banda ocupado (Hz)', 'Crest factor (Hz)', 'Valor RMS', 
                  'Frecuencia de Repetición de Pulso (PRF)', ])

    threshold = df['Power_dBm'].quantile(0.94) 

    # Encuentra las filas que superan el umbral
    df_above_threshold = df[df['Power_dBm'] > threshold]

    # Encuentra las regiones contiguas
    df_above_threshold['Region'] = (df_above_threshold['Frequency_Hz'].diff() > 1e6).cumsum()

    # Agrupa por las regiones identificadas
    regions = df_above_threshold.groupby('Region').agg(
        Min_Frequency=('Frequency_Hz', 'min'),
        Max_Frequency=('Frequency_Hz', 'max'),
        Peak_Power=('Power_dBm', 'max')
    ).reset_index(drop=True)

    for i in regions.index:
        df_filter = df[(df['Frequency_Hz']>=regions.loc[i, 'Min_Frequency'])&(df['Frequency_Hz']<=regions.loc[i, 'Max_Frequency'])]
        center_frequency = (df_filter['Frequency_Hz'].min() + df_filter['Frequency_Hz'].max()) / 2
        # Convertir la potencia de dBm a mW
        df_filter['Power_mW'] = 10 ** (df_filter['Power_dBm'] / 10)

        # Calcular el producto de frecuencia por potencia
        df_filter['Freq_Power'] = df_filter['Frequency_Hz'] * df_filter['Power_mW']

        # Calcular la frecuencia central ponderada
        weighted_center_frequency = df_filter['Freq_Power'].sum() / df_filter['Power_mW'].sum()

        # Encontrar la potencia máxima en dBm
        max_power_dBm = df_filter['Power_dBm'].max()

        # Calcular el umbral a -3 dB respecto a la potencia máxima
        threshold_dBm = max_power_dBm - 3

        # Filtrar las frecuencias donde la potencia es mayor o igual al umbral
        df_filter_filtered = df_filter[df_filter['Power_dBm'] >= threshold_dBm]

        # Calcular el ancho de banda
        bandwidth = df_filter_filtered['Frequency_Hz'].max() - df_filter_filtered['Frequency_Hz'].min()

        # Convertir la potencia de dBm a mW
        df_filter['Power_mW'] = 10 ** (df_filter['Power_dBm'] / 10)

        # Calcular la potencia total en mW
        total_power_mW = df_filter['Power_mW'].sum()

        # Calcular la potencia total de la señal en mW
        total_signal_power_mW = df_filter['Power_mW'].sum()

        # Supongamos que la potencia del ruido está dada en dBm
        noise_power_dBm = -30  # Ejemplo de potencia de ruido en dBm

        # Convertir la potencia del ruido a mW
        noise_power_mW = 10 ** (noise_power_dBm / 10)

        # Calcular la relación señal-ruido (SNR) en dB
        snr_dB = 10 * np.log10(total_signal_power_mW / noise_power_mW)

        # Detectar picos en la potencia
        peaks, _ = find_peaks(df_filter['Power_dBm'], height=threshold)  # Cambiar el valor de 'height' según sea necesario

        # Encontrar la potencia máxima
        max_power_dBm = df_filter['Power_dBm'].max()

        # Calcular la frecuencia de repetición de pulso (PRF)
        # Suponiendo que los picos corresponden a pulsos
        if len(peaks) > 1:
            # Calcular intervalos de tiempo entre picos
            peak_freqs = df_filter['Frequency_Hz'].iloc[peaks].values
            time_intervals = np.diff(1 / peak_freqs)  # Intervalos de tiempo entre pulsos
            prf = 1 / np.mean(time_intervals) if np.mean(time_intervals) > 0 else 0
        else:
            prf = 0

        # Convertir la potencia de dBm a mW para cálculos
        df_filter['Power_mW'] = 10 ** (df_filter['Power_dBm'] / 10)

        # Calcular la potencia promedio en mW
        average_power_mW = df_filter['Power_mW'].mean()

        # Convertir la potencia promedio de mW de vuelta a dBm
        average_power_dBm = 10 * np.log10(average_power_mW)

        # Calcular la potencia total
        total_power_mW = df_filter['Power_mW'].sum()

        # Calcular la potencia acumulativa
        df_filter['Cumulative_Power'] = df_filter['Power_mW'].cumsum()

        # Umbral para el 99% de la potencia total
        threshold_power = 0.97 * total_power_mW

        # Encontrar el rango de frecuencias que cumplen con el umbral
        lower_bound = df_filter[df_filter['Cumulative_Power'] >= (total_power_mW - threshold_power)].iloc[0]['Frequency_Hz']
        upper_bound = df_filter[df_filter['Cumulative_Power'] >= threshold_power].iloc[0]['Frequency_Hz']

        # Calcular el ancho de banda de ocupación
        bandwidth_occupation = upper_bound - lower_bound

        # Calcular la amplitud máxima y el valor RMS
        amplitude_max = df_filter['Power_mW'].max()
        value_rms = np.sqrt(np.mean(df_filter['Power_mW']**2))

        # Calcular el Crest Factor
        crest_factor = amplitude_max / value_rms

        # Calcular el nivel de ruido como el valor mínimo de la potencia en dBm
        noise_level_dBm = df_filter['Power_dBm'].min()

        # Convertir el nivel de ruido de dBm a mW
        noise_level_mW = 10 ** (noise_level_dBm / 10)

        # Calcular la amplitud (tomando la raíz cuadrada)
        df_filter['Amplitude'] = np.sqrt(df_filter['Power_mW'])
        
        frecuencia_fundamental = df_filter['Frequency_Hz'].min()
        harmonics = df_filter['Frequency_Hz'][df_filter['Frequency_Hz'] % frecuencia_fundamental == 0]/1000000
        frec_espurea = "< "+str(round(threshold, 2))
        frec_armonica = harmonics.to_string(index=False)
        modulacion = "Digital"

        lista = [center_frequency/1000000, weighted_center_frequency/1000000, bandwidth, average_power_mW, noise_level_mW, snr_dB,
                frec_espurea, frec_armonica, modulacion, len(peaks), bandwidth_occupation, crest_factor, value_rms, 
                prf]
        analysis_df.loc[len(analysis_df)] = lista

    peaks, _ = find_peaks(df['Power_dBm'], height=threshold) 
    # Crear gráficas (ajusta según las columnas de tu CSV)
    banda = px.line(df, x='Frequency_Hz', y='Power_dBm')
    banda.add_scatter(x=df['Frequency_Hz'].iloc[peaks], y=df['Power_dBm'].iloc[peaks],
                    mode='markers', marker=dict(color='red', size=10), name='Picos Espectrales')
    for i in regions.index:
        banda.add_vrect(x0=regions.loc[i, 'Min_Frequency'], x1=regions.loc[i, 'Max_Frequency'], fillcolor="green", opacity=0.25, line_width=0)

    banda.add_hline(y=threshold, line_color='black', line_dash='dash', annotation_text='Corte ruido')

    banda.update_layout(xaxis_title='Frecuencia (Hz)', yaxis_title='Potencia (dBm)')

    # Graficar las frecuencias espúreas
    espureas = px.scatter(df, 
                    x='Frequency_Hz', 
                    y='Power_dBm', 
                    labels={'Frequency_Hz': 'Frecuencia (Hz)', 'Power_dBm': 'Potencia (dBm)'},
                    color='Power_dBm',  # Color según potencia
                    color_continuous_scale=px.colors.sequential.Viridis)

    espureas.update_layout(xaxis_title='Frecuencia (Hz)', yaxis_title='Potencia (dBm)', 
                    showlegend=False)

    espectro = px.imshow(df3.T.sort_index(ascending=False), color_continuous_scale='Viridis', origin='lower',  
                         zmax=-68,zmin=-80,
                         height=800, 
                    labels={'x': 'Frecuencia', 'y': 'Tiempo', 'color': 'Potencia (dBm)'})
    res = analysis_df.round(2).T.reset_index()
    res.rename(columns={'index':'Variable'}, inplace=True)
    res.to_excel('../results.xlsx', index=False)
    columnas = []
    for col in res.columns:
        columnas.append(res[col])
    nueva_lista = [f"Señal {item}" if isinstance(item, int) else item for item in list(res.columns)]

    tabla = go.Figure(data=[go.Table(
            header=dict(values=nueva_lista,
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=columnas,
                    fill_color='lavender',
                    align='left'))
            ])
    tabla.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
    regions['Min_Frequency'] = regions['Min_Frequency']/1000000
    regions['Max_Frequency'] = regions['Max_Frequency']/1000000
    regions = regions.round(2)
    regions['Señal'] = regions.index
    
    regions = regions[['Señal', 'Min_Frequency', 'Max_Frequency', 'Peak_Power']]
    regions.rename(columns={'Min_Frequency':'Frecuencia mínima MHz', 'Max_Frequency':'Frecuencia máxima MHz', 'Peak_Power':'Pico de poder'}, inplace=True)
    columnas = []
    for col in regions.columns:
        columnas.append(regions[col])
    tabla2 = go.Figure(data=[go.Table(
            header=dict(values=regions.columns,
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=columnas,
                    fill_color='lavender',
                    align='left'))
            ])
    tabla2.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    texto = 'Se encontraron {} señales'.format(regions.shape[0])

    return f'Datos cargados: {len(df)} filas', banda, espureas, espectro, tabla, tabla2, texto

# Ejecutar la aplicación
#if __name__ == '__main__':
#    app.run_server(debug=True)

# Ejecutar la aplicación en un hilo separado para abrir el navegador
if __name__ == '__main__':
    app.run_server(port=8053, debug=True)
    
webbrowser.open_new("http://127.0.0.1:8053")


# %%
