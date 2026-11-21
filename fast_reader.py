"""
Lector de Patentes Argentinas - FAST-ALPR (Versión Oficial Documentada)
=======================================================================
Instalación: pip install "fast-alpr[onnx]" opencv-python numpy requests python-dotenv
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import cv2
import time
import threading
from datetime import datetime

import requests
from dotenv import load_dotenv
from fast_alpr import ALPR
import warnings
warnings.filterwarnings("ignore")

load_dotenv(r"F:\Repos\.env")

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
CAMERA_INDEX = "rtsp://LPR-camera:admin1234@192.168.1.45:554/stream1"
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 15   # > BURST_MAX_SECONDS a propósito, sino nunca frena nada (ver burst)

# Un auto acercándose genera varias lecturas seguidas (a veces con patentes mal leídas
# de fondo). Se acumulan en un "burst" y recién al cerrarse se decide y se envía una sola.
BURST_GAP_SECONDS = 1.5   # sin lecturas nuevas por este tiempo => el auto ya pasó, se cierra el burst
BURST_MAX_SECONDS = 3.0   # tope de seguridad si las lecturas no paran de llegar

ACCESS_SERVICE_URL = "http://localhost:8081/api/v1/access"
DEVICE_ID = "cam-patente-1"
HTTP_TIMEOUT_SECONDS = 3
HTTP_MAX_REINTENTOS = 1   # reintentos extra si falla el POST (0 = sin reintento)

# Keycloak (service account del client de esta cámara, credenciales en .env)
KEYCLOAK_TOKEN_URL = "http://localhost:8180/realms/Parko/protocol/openid-connect/token"
KEYCLOAK_CLIENT_ID = os.environ["KEYCLOAK_CLIENT_ID"]
KEYCLOAK_CLIENT_SECRET = os.environ["KEYCLOAK_CLIENT_SECRET"]
TOKEN_REFRESH_MARGIN_SECONDS = 10   # pide token nuevo si al actual le queda menos que esto

# ─────────────────────────────────────────────
#  VALIDACIÓN Y FORMATO (ARGENTINA)
# ─────────────────────────────────────────────
def formatear_patente_argentina(texto_crudo: str) -> tuple[bool, str, str]:
    texto = texto_crudo.upper().replace(' ', '').replace('-', '').replace('.', '')
    
    if len(texto) == 6 and texto[:3].isalpha() and texto[3:].isdigit():
        return True, "Viejo", f"{texto[:3]} {texto[3:]}"
    
    elif len(texto) == 7 and texto[:2].isalpha() and texto[2:5].isdigit() and texto[5:].isalpha():
        return True, "Mercosur", f"{texto[:2]} {texto[2:5]} {texto[5:]}"
    
    return False, "", texto

# ─────────────────────────────────────────────
#  SELECCIÓN DE PATENTE (BURST DE LECTURAS)
# ─────────────────────────────────────────────
def elegir_patente_ganadora(lecturas: list[dict]) -> tuple[str, str, float, int]:
    """Agrupa lecturas por texto exacto, promedia confianza por grupo y devuelve el de mayor promedio."""
    grupos: dict[str, list[dict]] = {}
    for lectura in lecturas:
        grupos.setdefault(lectura["texto"], []).append(lectura)

    mejor_texto, mejor_formato, mejor_promedio, mejor_cantidad = None, None, -1.0, 0
    for texto, items in grupos.items():
        promedio = sum(item["confianza"] for item in items) / len(items)
        if promedio > mejor_promedio:
            mejor_texto = texto
            mejor_formato = items[0]["formato"]
            mejor_promedio = promedio
            mejor_cantidad = len(items)

    return mejor_texto, mejor_formato, mejor_promedio, mejor_cantidad

# ─────────────────────────────────────────────
#  INTEGRACIÓN CON ACCESS-SERVICE
# ─────────────────────────────────────────────
_token_cache = {"access_token": None, "expires_at": 0.0}
_token_lock = threading.Lock()

def obtener_token() -> str:
    """Token de Keycloak (client_credentials, service account). Cacheado en memoria,
    se pide uno nuevo recién cuando falta poco para que expire el actual."""
    with _token_lock:
        if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - TOKEN_REFRESH_MARGIN_SECONDS:
            return _token_cache["access_token"]

        respuesta = requests.post(
            KEYCLOAK_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": KEYCLOAK_CLIENT_ID,
                "client_secret": KEYCLOAK_CLIENT_SECRET,
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        respuesta.raise_for_status()
        data = respuesta.json()
        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + data.get("expires_in", 60)
        return _token_cache["access_token"]

def enviar_a_access_service(patente: str):
    """POST a /api/access. Corre en un hilo aparte (ver procesar_burst_si_corresponde)
    para no trabar el loop de cámara si la red o el servicio están lentos/caídos.
    Reintenta HTTP_MAX_REINTENTOS veces si falla (timeout, servicio caído, token, etc.)."""
    payload = {
        "accessMethod": "PLATE",
        "identifier": patente,
        "deviceId": DEVICE_ID,
        "occurredAt": datetime.now().isoformat(),
    }

    intentos_totales = HTTP_MAX_REINTENTOS + 1
    for intento in range(1, intentos_totales + 1):
        try:
            headers = {"Authorization": f"Bearer {obtener_token()}"}
            respuesta = requests.post(ACCESS_SERVICE_URL, json=payload, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
            respuesta.raise_for_status()
            data = respuesta.json()

            if data.get("result") == "AUTHORIZED":
                print(f"BARRERA ABIERTA (simulado) - {data.get('eventType')} - {patente}")
            else:
                print(f"ACCESO DENEGADO (simulado) - {patente} - {data.get('message')}")
            return

        except requests.exceptions.RequestException as e:
            if intento < intentos_totales:
                print(f"Fallo intento {intento}/{intentos_totales} contra access-service, reintentando: {e}")
            else:
                print(f"No se pudo contactar a access-service tras {intentos_totales} intento(s) "
                      f"({ACCESS_SERVICE_URL}): {e}")

def procesar_burst_si_corresponde(lecturas_burst: list[dict], registro_cooldown: dict) -> list[dict]:
    """Si el burst sigue "vivo" (lecturas recientes) lo deja acumular. Si se cerró, decide
    la patente ganadora y dispara el envío a access-service."""
    if not lecturas_burst:
        return lecturas_burst

    ahora = time.time()
    tiempo_desde_ultima = ahora - lecturas_burst[-1]["timestamp"]
    tiempo_desde_primera = ahora - lecturas_burst[0]["timestamp"]

    if tiempo_desde_ultima < BURST_GAP_SECONDS and tiempo_desde_primera < BURST_MAX_SECONDS:
        return lecturas_burst

    patente, formato, confianza_promedio, cantidad = elegir_patente_ganadora(lecturas_burst)

    ahora_dt = datetime.now()
    ultima_vez = registro_cooldown.get(patente)
    if ultima_vez is None or (ahora_dt - ultima_vez).seconds >= COOLDOWN_SECONDS:
        print(f"Patente elegida tras {cantidad} lectura(s): {patente} "
              f"({formato}, confianza promedio {confianza_promedio*100:.1f}%)")
        threading.Thread(target=enviar_a_access_service, args=(patente,), daemon=True).start()
        registro_cooldown[patente] = ahora_dt

    return []

# ─────────────────────────────────────────────
#  HILO DE INTELIGENCIA ARTIFICIAL
# ─────────────────────────────────────────────
estado_app = {
    "frame_actual": None,
    "detecciones": [],
    "corriendo": True,
    "lock": threading.Lock()
}

def trabajador_ia(alpr_motor):
    while estado_app["corriendo"]:
        frame = None
        
        with estado_app["lock"]:
            if estado_app["frame_actual"] is not None:
                frame = estado_app["frame_actual"]
                estado_app["frame_actual"] = None 

        if frame is None:
            time.sleep(0.01)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            resultados = alpr_motor.predict(frame_rgb)
            nuevas_detecciones = []

            for res in resultados:
                # Accedemos a la estructura correcta según el debug
                if res.ocr is not None and res.detection is not None:
                    # La confianza del texto es una lista (una por letra). Sacamos el promedio.
                    if isinstance(res.ocr.confidence, list) and len(res.ocr.confidence) > 0:
                        confianza_texto = sum(res.ocr.confidence) / len(res.ocr.confidence)
                    else:
                        confianza_texto = res.ocr.confidence

                    texto = res.ocr.text
                    
                    # Filtramos por confianza (usamos la confianza del texto)
                    if confianza_texto >= CONFIDENCE_THRESHOLD and texto:
                        # Obtenemos las coordenadas desde detection.bounding_box
                        x1 = res.detection.bounding_box.x1
                        y1 = res.detection.bounding_box.y1
                        x2 = res.detection.bounding_box.x2
                        y2 = res.detection.bounding_box.y2
                        
                        w, h = x2 - x1, y2 - y1
                        
                        nuevas_detecciones.append({
                            "texto": texto,
                            "confianza": confianza_texto,
                            "x": int(x1), "y": int(y1), "w": int(w), "h": int(h)
                        })
            
            with estado_app["lock"]:
                estado_app["detecciones"] = nuevas_detecciones

        except Exception as e:
            print(f"Error interno en la IA: {e}")
            time.sleep(1)

# ─────────────────────────────────────────────
#  LOOP PRINCIPAL (CÁMARA Y DIBUJADO)
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  LECTOR DE PATENTES 2.0 (Motor FAST-ALPR)")
    print("=" * 55)
    print("Inicializando motor de IA...")

    # NUEVA FORMA DE INICIALIZAR SEGÚN EL README OFICIAL
    alpr_motor = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model"
    )
    print("Motor listo.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: No se encontró cámara en el índice {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hilo_ia = threading.Thread(target=trabajador_ia, args=(alpr_motor,), daemon=True)
    hilo_ia.start()

    registro_cooldown = {}
    lecturas_burst = []
    print("Cámara activa. Presioná 'Q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        with estado_app["lock"]:
            if estado_app["frame_actual"] is None:
                estado_app["frame_actual"] = frame.copy()
            detecciones_actuales = estado_app["detecciones"]

        for det in detecciones_actuales:
            texto_crudo = det["texto"]
            es_valida, formato, patente_limpia = formatear_patente_argentina(texto_crudo)

            if not es_valida:
                continue

            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            conf = det["confianza"]

            color = (0, 255, 100)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)

            label = f"PATENTE: {patente_limpia}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (x, y - th - 12), (x + tw + 8, y), color, -1)
            cv2.putText(display, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            lecturas_burst.append({
                "texto": patente_limpia,
                "formato": formato,
                "confianza": conf,
                "timestamp": time.time(),
            })

        lecturas_burst = procesar_burst_si_corresponde(lecturas_burst, registro_cooldown)

        cv2.imshow("Lector de Patentes", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    estado_app["corriendo"] = False
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema cerrado.")

if __name__ == "__main__":
    main()