import cv2

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara. ¿Permisos en Windows?")
        return

    print("Presiona 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: No se pudo leer frame.")
            break

        cv2.imshow("Test Cam - OpenCV", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
