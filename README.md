# NFCorpus API en Render

1. Sube `app.py`, `requirements.txt`, `Dockerfile` y `render.yaml` a la raíz de un repositorio Git conectado a Render. Puedes conservarlo privado.
2. En Render, selecciona **New → Blueprint** y elige el repositorio. La configuración crea un Web Service Docker en el plan Free con `/health` como comprobación de disponibilidad. Antes de aplicar, verifica que el plan sea Free.
3. Cuando el servicio indique **Live**, copia la URL `.onrender.com`. Abre `/health` y comprueba que `status` sea `ok` y `documents` sea `3633`.
4. Importa `NFCorpus.postman_collection.json`. En las variables de la colección cambia `base_url` por la URL pública, sin barra final. Prueba las solicitudes **Disponibilidad**, **Buscar con BM25** y **Buscar con TF-IDF**.

Para ejecutar en Windows con Python, instala `datasets` con `python -m pip install "datasets>=3.6,<6"` y ejecuta `python app.py`. La URL local es `http://127.0.0.1:7860`.

El servidor carga el corpus público `BeIR/nfcorpus` al iniciar. El primer arranque y el regreso desde inactividad pueden tardar mientras Render inicia el contenedor e indexa el corpus.
