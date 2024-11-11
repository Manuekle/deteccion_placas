from flask import Flask, render_template
import cv2
import pytesseract
import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

def capturar_imagen():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('static/car_image.jpg', frame)
    cap.release()
    return 'static/car_image.jpg'

def detectar_placa(filepath):
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    placa_contorno = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            placa_contorno = approx
            x, y, w, h = cv2.boundingRect(contour)
            placa = gray[y:y + h, x:x + w]
            break

    if placa_contorno is None:
        return "No se detectó ninguna placa"

    texto_placa = pytesseract.image_to_string(placa, config='--psm 8')  # PSM 8 es ideal para texto de una sola línea
    return texto_placa.strip()

def verificar_pico_y_placa(placa):
    ultimo_digito = int(placa[-1]) if placa and placa[-1].isdigit() else None
    dia_semana = datetime.datetime.today().weekday()

    restricciones = {
        0: [1, 2],  # Lunes: placas terminadas en 1 y 2
        1: [3, 4],  # Martes: placas terminadas en 3 y 4
        2: [5, 6],  # Miércoles: placas terminadas en 5 y 6
        3: [7, 8],  # Jueves: placas terminadas en 7 y 8
        4: [9, 0],  # Viernes: placas terminadas en 9 y 0
    }

    if ultimo_digito is None:
        return "No se pudo verificar la placa"
    if dia_semana in restricciones and ultimo_digito in restricciones[dia_semana]:
        return "No puede circular hoy"
    return "Puede circular hoy"

@app.route('/')
def index():
    imagen = capturar_imagen()
    placa = detectar_placa(imagen)
    estado_pico_placa = verificar_pico_y_placa(placa) if placa else "No se detectó ninguna placa"
    return render_template("result.html", placa=placa, estado_pico_placa=estado_pico_placa, imagen=imagen)

if __name__ == '__main__':
    app.run(debug=True)
