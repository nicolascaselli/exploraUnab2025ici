import gradio as gr
import numpy as np

EVENTO = "Jornada Explora UNAB – Taller ICI"
CAMPUS = "UNAB Concepción"

def passthrough(img: np.ndarray | None) -> np.ndarray:
    # img llega como (H,W,3) RGB cuando sources=["webcam"]
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return img

with gr.Blocks(title=f"{EVENTO} - {CAMPUS}", fill_height=True) as demo:
    gr.Markdown(f"### {EVENTO} — {CAMPUS}\n**Test webcam (Gradio 5)**\n"
                f"Permite la cámara en el navegador.")
    cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam", height=360)
    out = gr.Image(label="Salida", height=360)
    # En Gradio 5, debes especificar inputs/outputs en el stream:
    cam.stream(fn=passthrough, inputs=cam, outputs=out)

if __name__ == "__main__":
    # No uses demo.queue(...) con argumentos antiguos; podemos omitirlo.
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
