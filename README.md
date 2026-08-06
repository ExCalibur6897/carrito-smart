# Carrito Smart

Primer avance funcional de un carrito de compras inteligente. El prototipo combina
una interfaz de escritorio, inventario y ventas en SQLite, simulación manual de
eventos RFID y detección en vivo con un modelo YOLO nano preentrenado.

## Funcionalidad incluida

- Webcam en vivo con cajas de detección YOLO.
- Inferencia en un hilo de trabajo separado para no bloquear la interfaz.
- Lista de clases detectadas y porcentajes de confianza.
- Carrito con producto, cantidad, precio, subtotal y total.
- Entrada y salida manual de productos como sustituto temporal del RFID.
- Catálogo e inventario persistentes en SQLite.
- Pago aprobado simulado con confirmación del usuario.
- Registro de venta y descuento de inventario en una sola transacción SQLite.
- Reversión completa si algún producto ya no tiene stock suficiente.
- Logs rotativos de detecciones, carrito, errores y pagos.
- Seis productos iniciales para la demostración.

## Requisitos

- Windows 10/11
- Python 3.12 o 3.13 de 64 bits
- Webcam
- Conexión a Internet la primera vez que se descarguen los pesos nano (aprox. 5 MB)

No se utiliza ni se entrena un modelo personalizado. `yolo26n.pt` es el modelo
predeterminado y `yolo11n.pt` queda como fallback configurable. Ambos están
preentrenados con clases generales de COCO; demuestran la integración de visión,
pero todavía no identifican el catálogo específico de la tienda.

## Instalación

Desde PowerShell, en la raíz del proyecto:

```powershell
.\setup.ps1
```

El script busca Python 3.12, crea `.venv` e instala el proyecto con sus dependencias.
La alternativa manual es:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Ejecución

```powershell
.\run.ps1
```

También puede iniciarse directamente:

```powershell
.\.venv\Scripts\python.exe -m carrito_smart
```

La primera ejecución crea automáticamente:

- `data/carrito_smart.db`: catálogo, inventario y ventas.
- `data/Ultralytics/`: configuración local de Ultralytics.
- `logs/carrito_smart.log`: registro rotativo de actividad.

## Guion de demostración

1. Abra la aplicación y espere el estado **Webcam y YOLO activos**.
2. Muestre objetos comunes frente a la cámara y señale las cajas, clases y confianza.
3. Seleccione un producto y pulse **Entrada** dos veces.
4. Seleccione su fila y pulse **Salida** para simular retirar una unidad.
5. Observe que el total se recalcula y que el stock todavía no cambia.
6. Pulse **Simular pago aprobado** y confirme.
7. Muestre el número de venta; el carrito quedará vacío y el stock visible disminuirá.
8. Abra `logs/carrito_smart.log` si desea enseñar la trazabilidad.

Si la webcam no está disponible, la interfaz muestra el error y la parte de compra
sigue funcionando. Cierre Teams, Zoom, OBS u otra aplicación que pueda tener la
cámara ocupada y vuelva a iniciar Carrito Smart.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Las pruebas cubren datos iniciales, cálculo del carrito, salida manual, descuento
posterior al pago, rollback de una venta inválida, extracción de detecciones y
construcción de la ventana en modo sin pantalla.

## Configuración opcional

La aplicación acepta estas variables de entorno:

| Variable | Predeterminado | Uso |
| --- | --- | --- |
| `CARRITO_SMART_CAMERA` | `0` | Índice de webcam de OpenCV |
| `CARRITO_SMART_CAMERA_WIDTH` | `1280` | Ancho solicitado a la webcam |
| `CARRITO_SMART_CAMERA_HEIGHT` | `720` | Alto solicitado a la webcam |
| `CARRITO_SMART_YOLO_MODEL` | `yolo26n.pt` | Modelo principal o ruta a pesos |
| `CARRITO_SMART_FALLBACK_MODEL` | `yolo11n.pt` | Modelo usado si falla la carga principal |
| `CARRITO_SMART_CAMERA_FPS` | `30` | Máximo de capturas/video por segundo |
| `CARRITO_SMART_INFERENCE_FPS` | `8` | Máximo de inferencias YOLO por segundo |
| `CARRITO_SMART_UI_UPDATE_FPS` | `4` | Máximo de actualizaciones del texto por segundo |
| `CARRITO_SMART_DETECTION_THRESHOLD` | `0.60` | Umbral para aceptar una clase nueva |
| `CARRITO_SMART_RETENTION_THRESHOLD` | `0.45` | Umbral para mantener una clase confirmada |
| `CARRITO_SMART_CONFIRMATION_COUNT` | `3` | Detecciones consecutivas antes de confirmar |
| `CARRITO_SMART_DETECTION_HOLD_MS` | `700` | Retención tras una pérdida temporal |
| `CARRITO_SMART_CONFIDENCE_EMA_ALPHA` | `0.30` | Alpha de la media móvil de confianza |
| `CARRITO_SMART_INFERENCE_IMAGE_SIZE` | `640` | Tamaño de entrada de YOLO |
| `CARRITO_SMART_DATA_DIR` | `data` | Directorio de SQLite/configuración |
| `CARRITO_SMART_LOG_DIR` | `logs` | Directorio de logs |

Ejemplo para una segunda webcam:

```powershell
$env:CARRITO_SMART_CAMERA = "1"
.\run.ps1
```

## Arquitectura

```text
carrito_smart/
├── app.py             # arranque y composición
├── main_window.py     # interfaz PySide6
├── vision.py          # workers separados de webcam e inferencia
├── detection_stabilizer.py # confirmación, EMA, histéresis y retención
├── cart.py            # reglas del carrito
├── database.py        # esquema, consultas y transacción de venta
├── models.py          # modelos de dominio
├── config.py          # configuración por entorno
└── logging_config.py  # logs de consola y archivo
```

La UI no actualiza inventario directamente. El pago llama a `complete_sale`, que
abre `BEGIN IMMEDIATE`, vuelve a validar precio/stock, registra encabezado y detalle,
descuenta cada existencia y finalmente hace `COMMIT`. Cualquier error ejecuta
`ROLLBACK`.

## Estabilidad de visión y benchmark

La captura y la inferencia se ejecutan en dos `QThread` independientes y comparten
solamente el frame más reciente, sin una cola que acumule retraso. El video puede
mantener 30 FPS aunque YOLO opere a 8 FPS. Las cajas se dibujan en cada frame desde
el último estado confirmado, mientras que la lista de texto se limita a 4 FPS.

Para repetir la comparación con la misma cámara y configuración:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_vision.py `
  --duration 30 --models yolo11n.pt yolo26n.pt
```

El resultado se guarda en `logs/benchmarks/`. Las condiciones, métricas obtenidas y
la decisión de modelo están documentadas en
[`docs/VISION_STABILITY.md`](docs/VISION_STABILITY.md).
