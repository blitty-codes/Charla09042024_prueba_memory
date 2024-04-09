# Práctica

## Objetivo

El objetivo consiste en minimizar al máximo el cunsumo de RAM a la hora de utilizar el algoritmo del coseno.

## Premisas

- Los datos no tienen porque ser completamente frescos, es decir, se pueden generar archivos preprocesados que faciliten el cómputo del algoritmo.
- Solo se puede utilizar 1 núcleo.
- Los resultados obtenidos deben ser los mismos que los proporcionados por el script inicial.

## Pistas

- Entender y desglosar las diferentes partes del algoritmo del coseno para poder precalcular operaciones.
- Comprender y utilizar matrices dispersas con el objetivo de minimizar la memoria RAM utilizada.

## Preparación

### Descargar el repositorio de GIT

```bash
git clone https://github.com/diegoperezl/Charla09042024.git
```

Entrar en la carpeta del repositorio

```bash
cd Charla09042024
```

### Instalación del entorno

Usar el package manager [pip](https://pip.pypa.io/en/stable/) para instalar [pipenv](https://pipenv.pypa.io/en/latest/).

```bash
pip install pipenv
```

Instalar el entorno virtual.

```bash
pipenv install
```

### Ejecución

Acceder al entorno virtual.

```bash
pipenv shell
```

Ejecutar el servidor.

```bash
python app.py
```

Si se encuentran problemas con el puerto, cambiarlo al final del documento.

```bash
if __name__ == '__main__':
    app.run(host='localhost', port=9090, debug=False)
```

### Uso

Se pueden hacer peticiones a traves de aplicaciones como [postman](https://www.postman.com/), enviando en el cuerpo un JSON con el siguiente formato:

```bash
{
    "movie":1
}
```

También se pueden hacer peticiones a través de curl.

```bash
curl -i -H "Content-Type:application/json" -d "{\"movie\":1}" -X POST http://localhost:9090/cosine
```

### Uso de Custom

1. Generar los ficheros `python generate_files.py`
2. Ejecutar el servidor `python custom_app.py`
3. Probar con `curl -i -H "Content-Type:application/json" -d "{\"movie\":1}" -X POST http://localhost:9090/cosine`
