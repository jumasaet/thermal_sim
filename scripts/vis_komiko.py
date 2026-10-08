import glob
import os
import cv2
import numpy as np


def preprocess_thermal(data):
    """Normaliza los datos térmicos a uint8 (0-255) y aplica un mapa de calor."""
    # Eliminar dimensiones extra si vienen en formato (1, H, W) o (H, W, 1)
    data = np.squeeze(data)

    # Normalización Min-Max al rango 0 - 255
    min_val = np.min(data)
    max_val = np.max(data)

    if max_val > min_val:
        normalized = (data - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = np.zeros_like(data)

    img_uint8 = normalized.astype(np.uint8)

    # Aplicar mapa de color térmico (COLORMAP_INFERNO o COLORMAP_JET)
    colormap_img = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)

    return colormap_img, min_val, max_val


def view_thermal_dataset(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))

    if not files:
        print(f"No se encontraron archivos .npy en '{folder_path}'")
        return

    idx = 0
    window_name = "Visualizador Térmico"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Encontrados {len(files)} archivos.")
    print("Controles: [D/Flecha Derecha] Siguiente | [A/Flecha Izquierda] Anterior | [Q/ESC] Salir")

    while True:
        file_path = files[idx]
        data = np.load(file_path)

        display_img, t_min, t_max = preprocess_thermal(data)

        # Añadir texto con metadatos en pantalla
        filename = os.path.basename(file_path)
        info_text = f"[{idx+1}/{len(files)}] {filename} | Min: {t_min:.1f} | Max: {t_max:.1f}"
        cv2.putText(
            display_img,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(0) & 0xFF

        # Salir con 'q' o ESC (código 27)
        if key in (ord("q"), 27):
            break
        # Siguiente con 'd' o flecha derecha (código 83 o estándar según SO)
        elif key in (ord("d"), 83):
            idx = (idx + 1) % len(files)
        # Anterior con 'a' o flecha izquierda (código 81)
        elif key in (ord("a"), 81):
            idx = (idx - 1) % len(files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Sustituye por la ruta a tu carpeta con los .npy
    CARPETA_NPY = "/home/jumasaet/multimodal_sim/thermal_sim/examples/working/thermal"
    view_thermal_dataset(CARPETA_NPY)