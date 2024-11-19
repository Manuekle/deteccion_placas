from flask import Flask, render_template, Response, request
import cv2
import pytesseract
import datetime
import os

# Configura la ruta de Tesseract si estás en Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # Inicializa la cámara

# Ruta estática para guardar imágenes capturadas
CAPTURE_PATH = 'static/captures'
os.makedirs(CAPTURE_PATH, exist_ok=True)

def generar_video():
    """Genera frames de video en tiempo real."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Ruta para el video en tiempo real."""
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Página principal con video y botón de captura."""
    return render_template('index.html')

@app.route('/capturar', methods=['POST'])
def capturar():
    """Captura una imagen de la cámara y procesa la placa."""
    success, frame = camera.read()
    if success:
        # Define la ruta completa para guardar la imagen
        filepath = os.path.join(CAPTURE_PATH, 'captura.jpg')
        cv2.imwrite(filepath, frame)
        
        # Procesa la imagen para detectar la placa
        placa = detectar_placa(filepath)
        estado = verificar_pico_y_placa(placa) if placa else "No se detectó ninguna placa"
        
        # Devuelve los resultados a la plantilla
        return render_template('resultado.html', placa=placa, estado=estado, imagen=filepath)
    else:
        return "Error al capturar la imagen"

def detectar_placa(filepath):
    """Detecta la placa en la imagen proporcionada."""
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    placa = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            placa = gray[y:y + h, x:x + w]
            break

    if placa is None:
        return None
    return pytesseract.image_to_string(placa, config='--psm 8').strip()

def verificar_pico_y_placa(placa):
    """Verifica las restricciones de 'pico y placa' para la placa detectada."""
    ultimo_digito = int(placa[-1]) if placa and placa[-1].isdigit() else None
    dia_semana = datetime.datetime.today().weekday()

    restricciones = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7, 8],
        4: [9, 0],
    }

    if ultimo_digito is None:
        return "No se pudo verificar la placa"
    if dia_semana in restricciones and ultimo_digito in restricciones[dia_semana]:
        return "No puede circular hoy"
    return "Puede circular hoy"

if __name__ == '__main__':
    app.run(debug=True)
