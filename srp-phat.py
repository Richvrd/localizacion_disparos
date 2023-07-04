# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from random import uniform
from pyroomacoustics import doa, ShoeBox
from scipy.io import wavfile
import soundfile as sf
import glob
import time

def calc_ae(a,b):
    x = np.abs(a-b)
    return np.min(np.array((x, np.abs(360-x))), axis=0)

MAE, MEDAE, std_dev = {}, {}, {}
tiempo_for1 = time.time()
tiempo_for2 = time.time()

audio_files = glob.glob("uma8/*.wav")   #Importar audios

audio_data_list = []
sample_rate_list = []

for audio_file in audio_files:
    # Cargar el archivo de audio
    audio_data, sample_rate = sf.read(audio_file)
    
    # Agregar los datos de audio y la tasa de muestreo a las listas
    audio_data_list.append(audio_data)
    sample_rate_list.append(sample_rate)

# constants / config
fs = sample_rate_list[0]
nfft = 1024
n_frames = 30
max_order = 10
doas_deg = np.linspace(start=0, stop=359, num=360, endpoint=True)
# rs = [0.5, 1, 1.5]    #Distancias entre fuentes de sonido y micrófonos
mic_center = np.c_[[2,2,1]]
# mic_locs = mic_center + np.c_[[ 0.04,  0.0, 0.0],[ 0.0,  0.04, 0.0],
#                               [-0.04,  0.0, 0.0],[ 0.0, -0.04, 0.0],
# ]

# Configuración UMA-8
mic_distance = 0.0855  # Distancia entre los micrófonos (en metros)
num_mics = 7
radius = mic_distance * num_mics / (2 * np.pi)

angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
mic_positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.ones(num_mics)))
mic_positions = np.vstack((mic_positions, [0, 0, 0]))  # Agregar posición del micrófono central
mic_positions = mic_positions.T

# Configuraciones habitaciónes
room_dims = [[9,6,2.7],[5,3,2.7],[12,6,2.7]]
snr_lb, snr_ub = -5, 20

# room simulation
data = []
# for r in rs: //distancias entre la fuente y los micrófonos
for i, doa_deg in enumerate(doas_deg):
    doa_rad = np.deg2rad(doa_deg)
    # source_loc = mic_center[:,0] + np.c_[r*np.cos(doa_rad), r*np.sin(doa_rad), 0][0] //Ubicación con "r"
    source_loc = mic_center[:,0] + np.c_[np.cos(doa_rad), np.sin(doa_rad), 0][0]
    
    room_dim = room_dims[0]   #primera habitación
    # room_dim = room_dims[1] #segunda habitación
    # room_dim = room_dims[2] #tercera habitación

    room = ShoeBox(room_dim, fs=fs, max_order=max_order)
    room.add_source(source_loc, signal=audio_data_list[6].T)    #Agregar fuente (disparo)
    room.add_microphone_array(mic_positions)    #Agregar microfonos (uma8)
    room.simulate(snr=(snr_lb, snr_ub))     #Simular (snr(-5,20))
    signals = room.mic_array.signals    #360°, 8 señales

    # calculate n_frames stft frames starting at 1 second
    stft_signals = stft(signals[:,fs:fs+n_frames*nfft], fs=fs, nperseg=nfft, noverlap=0, boundary=None)[2]
    data.append([doa_deg, stft_signals])
    # data.append([r, doa_deg, stft_signals])
    
# tiempo_ejecucion1 = time.time() - tiempo_for1
# print("Tiempo de ejecución entorno: ",tiempo_ejecucion1)

kwargs = {'L': mic_positions,'fs': fs, 'nfft': nfft,'azimuth': np.deg2rad(np.arange(360))}
algorithms = {
    # 'MUSIC': doa.music.MUSIC(**kwargs),
    'SRP' : doa.srp.SRP(**kwargs),
}
# columns = ["r", "DOA"] + list(algorithms.keys())
columns = ["DOA"] + list(algorithms.keys())

predictions = {n:[] for n in columns}
# for r, doa_deg, stft_signals in data:
for doa_deg, stft_signals in data:
    # predictions['r'].append(r)
    predictions['DOA'].append(doa_deg)
    for algo_name, algo in algorithms.items():
        algo.locate_sources(stft_signals)   #Localizacion de fuente original vs fuente estimada por srp-phat
        predictions[algo_name].append(np.rad2deg(algo.azimuth_recon[0]))
df = pd.DataFrame.from_dict(predictions)

# tiempo_ejecucion2 = time.time() - tiempo_for2
# tiempo = tiempo_ejecucion2 - tiempo_ejecucion1


for algo_name in algorithms.keys():
    ae = calc_ae(df.loc[:,["DOA"]].to_numpy(), df.loc[:,[algo_name]].to_numpy())
    # Calculo MAE y MEDAE
    MAE[algo_name] = np.mean(ae)
    MEDAE[algo_name] = np.median(ae)
    # Cálculo de medidas de dispersión (desviación estándar)
    std_dev[algo_name] = np.std(ae)


for algo_name in algorithms.keys():
    print(f"Desviación estándar {algo_name}: {std_dev[algo_name]:5.2f}")

# print(f"MAE\t MUSIC: {MAE['MUSIC']:5.2f}\t NormMUSIC: {MAE['NormMUSIC']:5.2f} \t SRP: {MAE['SRP']:5.2f}")
# print(f"MEDAE\t MUSIC: {MEDAE['MUSIC']:5.2f}\t NormMUSIC: {MEDAE['NormMUSIC']:5.2f} \t SRP: {MEDAE['SRP']:5.2f}")
# SRP-PHAT
print(f"MAE\t SRP: {MAE['SRP']:5.2f}\t")
print(f"MEDAE\t SRP: {MEDAE['SRP']:5.2f}\t")
# MUSIC
# print(f"MAE\t MUSIC: {MAE['MUSIC']:5.2f}\t")
# print(f"MEDAE\t MUSIC: {MEDAE['MUSIC']:5.2f}\t")
# NormMUSIC
# print(f"MAE\t NormMUSIC: {MAE['NormMUSIC']:5.2f}\t")
# print(f"MEDAE\t NormMUSIC: {MEDAE['NormMUSIC']:5.2f}\t")
print("Tiempo de ejecución algoritmos: ",tiempo)
# if(cont==0):
#     print("Tiempo de ejecución algoritmos: ",tiempo)
# if(cont>0):
#     print("Tiempo de ejecución algoritmos: ",t2[cont]-t2[cont-1])
# print("------------------------------------------------------------------\n")
# cont = cont + 1
    
    
'''
De aquí para abajo van los graficos
'''

# # Crear una lista con los nombres de los algoritmos
# algorithms = ['MUSIC', 'NormMUSIC', 'SRP']

# # Crear una lista con los valores de MAE, MEDAE y Desviación estándar para cada algoritmo
# mae_values = [MAE['MUSIC'], MAE['NormMUSIC'], MAE['SRP']]
# medae_values = [MEDAE['MUSIC'], MEDAE['NormMUSIC'], MEDAE['SRP']]
# std_values = [std_dev['MUSIC'], std_dev['NormMUSIC'], std_dev['SRP']]

# # Configurar el gráfico
# plt.figure(figsize=(10, 6))
# plt.plot(algorithms, mae_values, marker='o', label='MAE')
# plt.plot(algorithms, medae_values, marker='o', label='MEDAE')
# plt.plot(algorithms, std_values, marker='o', label='Desviación estándar')

# # Personalizar el gráfico
# plt.xlabel('Algoritmo')
# plt.ylabel('Valor')
# plt.title('Comparación de MAE, MEDAE y Desviación estándar')
# plt.legend()

# # Mostrar el gráfico
# plt.show()



'''
__Graficos por separado__
'''
# # Obtener los nombres de los algoritmos
# algo_names = list(algorithms.keys())

# # Obtener los valores de MAE, MEDAE, desviación estándar y varianza
# mae_values = [MAE[algo_name] for algo_name in algo_names]
# medae_values = [MEDAE[algo_name] for algo_name in algo_names]
# std_dev_values = [std_dev[algo_name] for algo_name in algo_names]
# variance_values = [variance[algo_name] for algo_name in algo_names]

# # Crear una figura con subplots
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# # Graficar el MAE
# axs[0, 0].bar(algo_names, mae_values)
# axs[0, 0].set_title('MAE')
# axs[0, 0].set_ylabel('Valor')
# axs[0, 0].set_xlabel('Algoritmo')

# # Graficar el MEDAE
# axs[0, 1].bar(algo_names, medae_values)
# axs[0, 1].set_title('MEDAE')
# axs[0, 1].set_ylabel('Valor')
# axs[0, 1].set_xlabel('Algoritmo')

# # Graficar la desviación estándar
# axs[1, 0].bar(algo_names, std_dev_values)
# axs[1, 0].set_title('Desviación estándar')
# axs[1, 0].set_ylabel('Valor')
# axs[1, 0].set_xlabel('Algoritmo')

# # Graficar la varianza
# axs[1, 1].bar(algo_names, variance_values)
# axs[1, 1].set_title('Varianza')
# axs[1, 1].set_ylabel('Valor')
# axs[1, 1].set_xlabel('Algoritmo')

# # Ajustar los espacios entre subplots
# plt.tight_layout()

# # Mostrar la figura
# plt.show()