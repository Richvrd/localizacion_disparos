import numpy as np
import soundfile as sf
from pyroomacoustics import *
from pyroomacoustics.doa import circ_dist
import matplotlib.pyplot as plt

# Configuración de los parámetros
mic_distance = 0.0855  # Distancia entre los micrófonos (en metros)
fs = 44100  # Frecuencia de muestreo de los archivos de audio

# Crear la geometría del array de micrófonos UMA-8
num_mics = 7    
radius = mic_distance * num_mics / (2 * np.pi)

angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
mic_positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.ones(num_mics)))
mic_positions = np.vstack((mic_positions, [0, 0, 0]))  # Agregar posición del micrófono central

print(mic_positions)

fig, ax = plt.subplots()
ax.plot(mic_positions[:, 0], mic_positions[:, 1], 'o', label='Micrófonos')
ax.plot(0, 0, 'ro', label='Micrófono central')  # Micrófono central en el origen (0, 0)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Array de micrófonos UMA-8')
ax.legend()
plt.show()