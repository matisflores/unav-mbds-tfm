# Trabajo Final de Master
El presente programa fue generado como soporte a la investigación titulada **TRACKING DE OBJETOS Y CARACTERIZACIÓN DE ACTIVIDAD MEDIANTE COMPUTER VISION** correspondiente al Master en Big Data Science de la Universidad de Navarra.

![App Platorm](/assets/frame.jpg)

## Dependencias
```
pip install -r requirements.txt
```

## Configuración
```
[general]
# video to be processed
source = data/people.mp4

# output directory
data_dir = data

# adjust cell size for grid matrix
cell_size = 40

# enable or disable use of region of interest
use_roi = true | empty (false)

# only make detections on frame_number % detection_rate == 0
detection_rate = 2

# downscale video to increase velocity
video_downscale = 1

# mark detections and trackers on each processed frame
show_detections = True
show_trackers = True

# tracker to use
tracker = gh | kalman | unscented | particle

# iou threshold
min_iou = 0.5

[yolo]
confidence_threshold = 0.5
```

## Ejecución
```
python src/main.py
```

### memory profiler
```
mprof run src/main.py
mprof plot
````

## Procesamiento de resultados
notebook disponible en [Google Colab](https://colab.research.google.com/drive/18QWi1jdQVh6h8hyZfJHFj0moUrUrtbqy)