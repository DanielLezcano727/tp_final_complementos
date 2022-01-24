#! /usr/bin/python

# Cavagna - Lezcano
# Complementos Matematicos I
# TP FINAL - Dibujar grafos


#------------------------<<Librerias importadas>>-----------------------------#
import argparse
import matplotlib.pyplot as plt
import numpy as np
#-----------------------------------------------------------------------------#

#------------------------<<Constantes>>-----------------------------#

C_REPULSION = 0.1
C_ATRACCION = 0.9

GRAFO_COMPLEJO = 20     #Cantidad de vertices
# Constantes para el cilindro: C_REPULSION: 0.15
# Constantes para el cilindro: C_ATRACCION: 0.1

# Constantes para los parametros por defecto
CONS_ANCHURA = 100
CONS_ALTURA = 100
CONS_REFRESH = 7
CONS_GRAVEDAD = 10
CANT_CONS_ITERS=100
CONS_TEMP = 0.95
CONS_TEMP_INI= 10.0
EPSILON= 0.005
#-----------------------------------------------------------------------------#



#------------------------<<Funciones auxiliares>>-----------------------------#
def sum_t(t1, t2):
    """
    Entrada: recibe dos tuplas de numeros reales

    Funcionamiento: realiza una suma de tuplas componente a componente

    salida:devuelve la suma de las tuplas
    """
    return (t1[0] + t2[0], t1[1] + t2[1])

def sub_t(t1, t2):
    """
    Entrada: recibe dos tuplas de numeros reales

    Funcionamiento: realiza una diferencias de tuplas componente a componente

    salida:devuelve la suma de las tuplas
    """
    return (t1[0] - t2[0], t1[1] - t2[1])



def lee_grafo_archivo(file_path):
    """
    Entrada: recibe la ruta de un archivo

    Funcionamiento: levanta un grafo desde un archivo

    salida:devuelve el grafo del archivo
    """
    with open(file_path, "r") as f:
        aristas = []
        cant_v=int(f.readline().rstrip())#rstrip sacatodo los espacios
        vertices = [f.readline().strip("\n") for i in range(cant_v)]

        aristas = [tuple(line.strip().split(' ')) for line in f]
        for v1, v2 in aristas:
            if v1 not in vertices or v2 not in vertices:
                raise Exception("vertice incorrecto")

    return (vertices, aristas)

#-----------------------------------------------------------------------------#



class LayoutGraph:

    def __init__(self, grafo, iters, refresh, c1, c2, altura, anchura, temp, gravedad, t0, verbose=False):
        """
        Parámetros:
        grafo: grafo en formato lista
        iters: cantidad de iteraciones a realizar
        refresh: cada cuántas iteraciones graficar. Si su valor es cero, entonces debe graficarse solo al final.
        c1: constante de repulsión
        c2: constante de atracción
        verbose: si está encendido, activa los comentarios
        """
        if verbose:
            print("Inicializando parametros del grafo")

        # Guardo el grafo
        self.grafo = grafo

        # Inicializamos las posiciones de los vertices
        self.posiciones = self.init_posiciones(grafo[0])

        # Guardo opciones
        self.iters = iters
        self.verbose = verbose
        self.altura = altura
        self.anchura = anchura
        self.gravedad = gravedad
        self.refresh = refresh
        self.t0 = t0

        # Calculamos las constantes de atraccion y repulsion
        self.cantidad_vertices = len(self.grafo[0])
        k = np.sqrt((anchura*altura)/self.cantidad_vertices)

        self.temperatura = temp
        self.k1 = c1*k
        self.k2 = c2*k


    def init_posiciones(self, vertices):
        """
        Entrada: recibe los vertices del grafo

        Funcionamiento: inicializa las posiciones de los vertices del grafo

        salida: devuelve un diccionario con los vertices y sus posiciones
        """
        if self.verbose:
            print("Inicializando posiciones de los vertices")

        return {
            key: tuple(np.random.randint(self.anchura, size=2))
            for key in vertices
        }

    def dibujar_grafo(self):
        """
        Entrada: none

        Funcionamiento: dibuja el grafo usando las funciones de matplotlib

        salida: none
        """
        if self.verbose:
            print("Graficando")

        # Limpiamos el grafico
        plt.clf()

        # Iniciamos variables
        minimo_y = self.altura
        maximo_y = 0
        minimo_x = self.anchura
        maximo_x = 0

        # Dibujamos los vertices y aristas en pantalla
        for v1 , v2 in self.grafo[1]:
            plt.plot(
                (self.posiciones[v1][0], self.posiciones[v2][0]),
                (self.posiciones[v1][1], self.posiciones[v2][1]),
                'g',
                marker='o',
                mfc='black'
            )

        # Les ponemos nombre y calculamos el vertice que esta mas cerca de los bordes
        for v in self.grafo[0]:
            plt.annotate(v, xy=(self.posiciones[v][0], self.posiciones[v][1] + 2))
            
            maximo_y = max(self.posiciones[v][1], maximo_y)
            maximo_x = max(self.posiciones[v][0], maximo_x)
            minimo_y = min(self.posiciones[v][1], minimo_y)
            minimo_x = min(self.posiciones[v][0], minimo_x)

        # Dibujamos los ejes cercanos al objeto y pausamos el dibujo            
        plt.axis((minimo_x - 5, maximo_x + 5, minimo_y - 5, maximo_y + 5))
        plt.pause(0.041)


    def layout(self):
        """
        Entrada: none

        Funcionamiento: aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
        un layout

        salida: none
        """

        if self.verbose:
            print("Iniciando graficado")

        # Variables
        const = 0

        for it in range(0,self.iters):
            # Mostramos las iteraciones cada self.refresh calculos
            if const*self.refresh == it:
                self.dibujar_grafo()
                const+=1
            self.step()

        # Mostramos el dibujo
        plt.show()


    def step(self):
        """
        Entrada: none

        Funcionamiento: lleva a cabo los calculos de las fuerzas de atraccion, repulsion , gravedad y actualiza las 
        posiciopnes 

        salida: none
        """
        accum = self.init_acumuladores(self.grafo[0])
        self.computar_fuerzas_atraccion(accum)
        self.computar_fuerzas_repulsion(accum)
        self.computar_gravedad(self.grafo[0], accum)
        self.actualizar_posiciones(self.grafo[0], accum)
        self.update_temperature()


    def update_temperature(self):
        """
        Entrada: none

        Funcionamiento: actualiza la temperatura a partir de t0

        salida: none
        """
        self.temperatura *= self.t0

    def computar_gravedad(self, vertices, accum):
        """
        Entrada: recibe los vertices del grafo y los acumuladores

        Funcionamiento: acumula la fuerza de gravedad de cada vertice en los acumuladores

        salida: none
        """
        if self.verbose:
            print("Computando fuerza de gravedad...")

        # Por cada vertice le aplicamos una fuerza al centro de la pantalla
        for vertice in vertices:
            x, y = self.posiciones[vertice]
            grav = (self.anchura / 2 - x, self.altura / 2 - y)
            distancia = np.linalg.norm(grav)

            if distancia > EPSILON:
                f = self.calc_f(self.gravedad, self.posiciones[vertice], (self.anchura / 2,self.altura / 2), distancia)
                accum[vertice] = sub_t(accum[vertice], f)

    def computar_fuerzas_atraccion(self, accum):
        """
        Entrada: recibe los acumuladores

        Funcionamiento: acumula las fuerzas de atraccion de cada vertice en los acumuladores

        salida: none
        """
        if self.verbose:
            print("Computando fuerzas de atraccion...")

        # Por cada par arista calculamos su atraccion
        for v1 , v2 in self.grafo[1]:
            distancia = self.norma(v1,v2)

            # Si estan muy cerca los alejamos
            if distancia < EPSILON:
                num=np.random.rand()
                self.posiciones[v2] = sum_t(self.posiciones[v2], (num, num))
                self.posiciones[v1] = sub_t(self.posiciones[v1], (num, num))
                distancia = self.norma(v1,v2)

            # Calculamos la atraccion y modificamos ambos vertices
            fuerza = self.calcular_atraccion(distancia)
            f = self.calc_f(fuerza, self.posiciones[v2], self.posiciones[v1], distancia)

            accum[v1] = sum_t(accum[v1], f)
            accum[v2] = sub_t(accum[v2], f)

    def computar_fuerzas_repulsion(self, accum):
        """
        Entrada: recibe los acumuladores

        Funcionamiento: acumula las fuerzas de repulsion de cada vertice en los acumuladores

        salida: none
        """
        if self.verbose:
            print("Computando fuerzas de repulsion...")

        # Por cada vertice que esta en el mismo cuadrante calculamos su repulsion
        for v1 in self.grafo[0]:
            for v2 in self.grafo[0]:
                if v1 != v2:
                    distancia = self.norma(v1,v2)
 
                    # Si estan muy cerca, los alejamos
                    if distancia<EPSILON:
                        num=np.random.rand()
                        self.posiciones[v2] = sum_t(self.posiciones[v2], (num, num))
                        self.posiciones[v1] = sub_t(self.posiciones[v1], (num, num))
                        distancia = self.norma(v1,v2)

                    fuerza = self.calcular_repulsion(distancia)
                    f = self.calc_f(fuerza, self.posiciones[v2], self.posiciones[v1], distancia)
                    accum[v1] = sub_t(accum[v1], f)
                    accum[v2] = sum_t(accum[v2], f)

    def actualizar_posiciones(self, vertices, accum):
        """
        Entrada: Vertices del grafo y las fuerzas de cada vertice acumuladas

        Funcionamiento: Aplica la fuerza acumulada a los vertices correspondientes. Tambien, actualiza la temperatura

        salida: None
        """
        if self.verbose:
            print("Actualizando posiciones")

        # Actualizamos la posicion de cada vertice
        for vertice in vertices:
            modulo = np.linalg.norm(accum[vertice])

            # Aplicamos el calculo de la temperatura
            if (modulo > self.temperatura):
                v_x, v_y = accum[vertice]
                f_x = (v_x / modulo) * self.temperatura
                f_y = (v_y / modulo) * self.temperatura
                accum[vertice] = (f_x, f_y)

            nueva_posicion = sum_t(self.posiciones[vertice], accum[vertice])
            self.posiciones[vertice] = self.limit_point(nueva_posicion)

    def norma(self, v1, v2):
        """
        Entrada: recibe dos vertices del grafo

        Funcionamiento: calcula la distancia euclidiana entre los vertices

        salida: devuelve dicha distancia 
        """
        x1, y1 = self.posiciones[v1]
        x2, y2 = self.posiciones[v2]
        return np.linalg.norm((x1 - x2, y1 - y2))

    def calcular_atraccion(self, distancia):
        """
        Entrada: Recibe la distancia entre los vertices

        Funcionamiento: Calcula la fuerza de atraccion. Si hay muchos vertices, aplica una formula para grafos complejos

        salida: La fuerza de atraccion
        """
        if self.cantidad_vertices <= GRAFO_COMPLEJO:
            return (distancia**2)/self.k1
        else:    
            return self.k1 * np.log(distancia)

    def calcular_repulsion(self, distancia):
        """ Aplica y devuelve la formula de la repulsion """
        return (self.k2**2)/distancia

    def init_acumuladores(self, vertices):
        """ Inicializa las fuerzas acumuladas de cada vertice """
        return { vertice: (0, 0) for vertice in vertices }

    def limit(self, n, lim):
        """ Si el numero esta entre 0 y el limite, devuelve el numero. Sino, devuelve 0 o lim, dependiendo de que limite infringio """
        return max(0, min(lim, n))

    def limit_point(self, p):
        """ Devuelve las coordenadas del punto limitada por los bordes de la pantalla """
        return (self.limit(p[0], self.anchura), self.limit(p[1], self.altura))

    def calc_f(self, f, v1, v2, distancia):
        """ calcula un vector entre dos puntos con modulo f """
        return (f * (v1[0] - v2[0]) / distancia, f * (v1[1] - v2[1]) / distancia)




def main():
    # Definimos los argumentos de linea de comando que aceptamos
    parser = argparse.ArgumentParser()

    # Verbosidad, opcional, False por defecto
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Muestra mas informacion al correr el programa'
    )
    # Cantidad de iteraciones, opcional, CANT_CONS_ITERS por defecto
    parser.add_argument(
        '--iters',
        type=int,
        help='Cantidad de iteraciones a efectuar',
        default=CANT_CONS_ITERS
    )
    
    #Anchura de la ventana, opcional, CONS_ANCHURA por defecto
    parser.add_argument(
        '--anchura',
        type=int,
        help='Anchura de la ventana(Natural)',
        default=CONS_ANCHURA
    )

    #Altura de la ventana, opcional, CONS_ALTURA por defecto
    parser.add_argument(
        '--altura',
        type=int,
        help='Altura de la ventana(Natural)',
        default=CONS_ALTURA
    )

    #Tasa de refrezco, opcional, CONS_REFRESH por defecto 
    parser.add_argument(
        '--refresh',
        type=int,
        help='Tasa de refresco de la graficadora(Natural)',
        default=CONS_REFRESH
    )


    # Temperatura inicial, opcional, CONS_TEMP_INI por defecto 
    parser.add_argument(
        '--temp',
        type=float,
        help='Temperatura inicial',
        default=CONS_TEMP_INI
    )


    # Constante para los calculos de atraccion, opcional, C_ATRACCION por defecto
    parser.add_argument(
        '--c1',
        type=float,
        help='Constante de atraccion',
        default=C_ATRACCION
    )

    # Constante para los calculos de repulsion, opcional, C_REPULSION por defecto
    parser.add_argument(
        '--c2',
        type=float,
        help='constante de repulsion',
        default=C_REPULSION
    )

    # Constante para la gravedad, opcional, C_GRAVEDAD por defecto
    parser.add_argument(
        '--gravedad',
        type=int,
        help='Constante de gravedad',
        default=CONS_GRAVEDAD
    )

    # Constante para disminuir la temperatura, opcional, CONS_TEMP por defecto
    parser.add_argument(
        '--t0',
        type=float,
        help='constante de cambio de temperatura',
        default=CONS_TEMP
    )

    # Archivo del cual leer el grafo
    parser.add_argument(
        'file_name',
        help='Archivo del cual leer el grafo a dibujar'
    )
    args = parser.parse_args()

    # # Creamos nuestro objeto LayoutGraph
    layout_gr = LayoutGraph(
         grafo = lee_grafo_archivo(args.file_name),
         iters=args.iters,
         refresh=args.refresh,
         c1=args.c1,
         c2=args.c2,
         altura=args.altura,
         anchura=args.anchura,
         verbose=args.verbose,
         temp=args.temp,
         gravedad=args.gravedad,
         t0=args.t0
     )
    
    # Ejecutamos el grafico
    layout_gr.layout()
    return


if __name__ == '__main__':
    main()
