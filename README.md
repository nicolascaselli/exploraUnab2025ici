# Taller Explora UNAB 2025 – Ingeniería Civil Informática

Este repositorio contiene los proyectos del **Taller de Visión por Computador con IA** realizado en la  
**Jornada Explora UNAB – Campus Concepción (2025)**.

Durante el taller, los estudiantes podrán elegir entre **dos proyectos** y, con ayuda de Inteligencia Artificial, modificar el código para completar **desafíos prácticos**.  
El objetivo es **simular el trabajo de un informático**, mostrando cómo la IA facilita la programación y permite crear aplicaciones divertidas en poco tiempo.

---

## 🚀 Proyectos disponibles

### 🖌️ Proyecto A: AirDraw
Dibuja en el aire con tu dedo índice frente a la cámara y observa cómo el trazo aparece en pantalla.

**Desafíos oficiales:**
1. **Borrador mágico:** al presionar la tecla `E`, el dedo dibuja en blanco (borrador).  
2. **Guardar dibujo completo:** al presionar la tecla `S`, se guarda todo el lienzo como imagen PNG en la carpeta `records_A/`.

Archivos relevantes:
- `opcion_A/airdraw_desktop.py`
- Carpeta de salida: `records_A/`

---

### 😀 Proyecto B: Face→Emoji
Detecta tu cara y muestra un emoji sobre tu rostro según gestos simples (sonrisa, sorpresa, neutral, etc.).

**Desafíos oficiales:**
1. **Texto del gesto:** mostrar debajo del emoji el texto correspondiente (“Sonrisa”, “Sorpresa”, “Triste”, etc.).  
2. **Emoji más grande en sorpresa:** cuando se detecte 😮, el emoji debe hacerse más grande que lo normal.

Archivos relevantes:
- `opcion_B/face_emoji.py`
- Carpeta de salida: `records_B/`

---

## 🖥️ Requisitos

- Windows 10/11 con cámara integrada o USB.
- [Python 3.12 (64 bits)](https://www.python.org/downloads/release/python-3120/) instalado y agregado al PATH.
- [PyCharm Community Edition](https://www.jetbrains.com/pycharm/download/) para abrir el proyecto.
- Conexión a internet para consultar la **IA asistente**.

Dependencias principales (instaladas en el entorno virtual `.venv`):
- `opencv-python==4.10.0.84`
- `mediapipe==0.10.14`
- `numpy==1.26.4`
- `pillow==10.3.0`

Dependencias adicionales (para plan B web con Gradio/Streamlit):
- `gradio==5.6.0`
- `streamlit==1.36.0`
- `streamlit-webrtc==0.47.1`
- `av==12.0.0`
- `aiortc==1.9.0`
- `aioice==0.9.0`

---

## 📥 Instalación rápida

1. **Clonar o descargar** este repositorio:  
   ```bash
   git clone https://github.com/nicolascaselli/exploraUnab2025ici
   cd exploraUnab2025ici
   ```

   > Alternativa: usar el botón **Code → Download ZIP** y descomprimir.

2. **Crear entorno virtual en Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Instalar dependencias principales:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## ▶️ Ejecución

### AirDraw
```powershell
.\.venv\Scripts\activate
python opcion_A\airdraw_desktop.py
```

### Face→Emoji
```powershell
.\.venv\Scripts\activate
python opcion_B\face_emoji.py
```

Al ejecutar se abrirá una ventana con la cámara en vivo:
- **AirDraw:** mueve el dedo índice para dibujar.  
- **Face→Emoji:** haz gestos (sonríe, sorpréndete, cierra los ojos).  

Los recuerdos (fotos automáticas) quedarán en las carpetas `records_A/` o `records_B/`.

---

## 🎯 Dinámica del taller

1. Forma un equipo y elige **Proyecto A (AirDraw)** o **Proyecto B (Face→Emoji)**.  
2. Ejecuta el código base y comprueba que funciona.  
3. Abre la IA (ChatGPT, Copilot, etc.) y plantea el **desafío** que quieres resolver.  
   - Ejemplo de prompt:  
     > “En el archivo `airdraw_desktop.py`, agrega que al presionar la tecla E se active un borrador que dibuje en blanco. Muéstrame el código exacto para pegar.”  
4. Copia y pega las modificaciones en PyCharm.  
5. Prueba en la cámara y guarda tus evidencias.  
6. **Sube tus recuerdos (imágenes PNG)** en la plataforma del taller:  
   👉 [https://explora2025.unabdevhub.cl](https://explora2025.unabdevhub.cl)

---

## 🏆 Recuerdos y premiación

- Cada equipo debe entregar al menos **2 imágenes** como evidencia:
  - Una mostrando la aplicación en acción.  
  - Una guardada automáticamente en `records_A/` o `records_B/`.  
- Las imágenes se subirán al sitio web y podrán verse en una **galería final** del taller.  
- Equipos que completen ambos desafíos recibirán un reconocimiento.

---

## 📚 Créditos

Taller diseñado y guiado por:  
**Dr. Nicolás Caselli Benavente**  
Director de Carrera Ingeniería Civil Informática  
Universidad Andrés Bello – Campus Concepción  

Con el apoyo del equipo de monitores de la carrera ICI.
