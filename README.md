# Trabajo Práctico final (Complementos de matemáticas 1)

Integrantes del trabajo:

    Lucas Gastón Cavagna
    Daniel Nicolas Lezcano


## Instalación de librerias:

Recomendamos utilizar un entorno virtual para no interferir con las librerias locales.

```
pip install argparse
pip install matplotlib
```

## Información de uso

Para una correcta ejecución del programa se debe proceder de la siguiente manera

```
python cavagna_lezcano.py <archivo_contiene_grafo>
```

Ejemplo

```
python cavagna_lezcano.py grafo.txt
```

## Parametros 

Se puede agregar algunos parametros al comando de ejecución del programa:

 - iters = Cantidad de iteraciones a efectuar  
 - verbose = Muestra información importante durante la ejecución del programa
 - altura = Altura de la ventana (Natural)
 - anchura = Anchura de la ventana (Natural)
 - gravedad = Constante de gravedad
 - refresh = Tasa de refresco de la graficadora(Natural)
 - t0 = constante de cambio de temperatura
 - temp= Temperatura inicial
 - c1 = Constante de atracción
 - c2 = constante de repulsión

Ejemplo de uso

python cavagna_lezcano.py grafo.txt --verbose --iters 100

## Aclaraciones

 - Hemos usado como fórmula para calcular la atracción la formula mencionada en el paper que usa el logaritmo ya que nos dio muchos mejores resultados
 - Al cambiar el tamaño de la pantalla con los parametros `anchura` y `altura` se deben tener en cuenta las constantes en general.
