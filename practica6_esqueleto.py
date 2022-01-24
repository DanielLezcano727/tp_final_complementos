#! /usr/bin/python

# 6ta Practica Laboratorio 
# Complementos Matematicos I
# Ejemplo parseo argumentos

import argparse
import matplotlib.pyplot as plt
import numpy as np

C_REPULSION = 0.1
C_ATRACCION = 0.9

MAX_X = 100
MAX_Y = 100

EPSILON= 0.005

def sum_t(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def sub_t(t1, t2):
    return (t1[0] - t2[0], t1[1] - t2[1])

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
        if self.verbose:
            print("Inicializando parametros del grafo")

        # Guardo el grafo
        self.grafo = grafo

        # Inicializo estado
        # Completar
        self.posiciones = {}

        # Guardo opciones
        self.iters = iters
        self.verbose = verbose
        self.altura = altura
        self.anchura = anchura
        self.gravedad = gravedad
        self.refresh = refresh
        self.t0 = t0

        k = np.sqrt((anchura*altura)/len(grafo[0]))

        self.temperatura = temp
        self.k1 = c1*k
        self.k2 = c2*k


    def init_posiciones(self, vertices):
        if self.verbose:
            print("Inicializando posiciones de los vertices")

        return {
            key: tuple(np.random.randint(101,size=2))
            for key in vertices
        }

    def dibujar_grafo(self):
        if self.verbose:
            print("Graficando")

        # Limpiamos el grafico
        plt.clf()

        # Iniciamos variables
        minimo_y = self.altura
        maximo_y = 0
        minimo_x = self.anchura
        maximo_x = 0

        # Dibujamos los vertices en pantalla
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
            plt.annotate(v, xy=(self.posiciones[v][0], self.posiciones[v][1] + 5))
            
            maximo_y = max(self.posiciones[v][1], maximo_y)
            maximo_x = max(self.posiciones[v][0], maximo_x)
            minimo_y = min(self.posiciones[v][1], minimo_y)
            minimo_x = min(self.posiciones[v][0], minimo_x)

        # Dibujamos los ejes cercanos al objeto y pausamos el dibujo            
        plt.axis((minimo_x - 20, maximo_x + 20, minimo_y - 20, maximo_y + 20))
        plt.pause(0.1)


    def layout(self):
        """
        Aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
        un layout
        """
        if self.verbose:
            print("Iniciando graficado")

        # Variables
        const = 0
        self.posiciones = self.init_posiciones(self.grafo[0])

        for it in range(0,self.iters):
            # Mostramos las iteraciones cada self.refresh calculos
            if const*self.refresh == it:
                self.dibujar_grafo()
                const+=1
            self.step()

        # Mostramos el dibujo
        plt.show()


    def step(self):
        accum = self.init_acumuladores(self.grafo[0])
        self.computar_fuerzas_atraccion(accum)
        self.computar_fuerzas_repulsion(accum)
        self.computar_gravedad(self.grafo[0], accum)
        self.actualizar_posiciones(self.grafo[0], accum)
        self.update_temperature()

    def update_temperature(self):
        self.temperatura *= 0.95

    def computar_gravedad(self, vertices, accum):
        if self.verbose:
            print("Computando fuerza de gravedad...")

        # Por cada vertice le aplicamos una fuerza al centro de la pantalla
        for vertice in vertices:
            x, y = self.posiciones[vertice]
            grav = (self.anchura / 2 - x, self.altura / 2 - y)
            distancia = np.linalg.norm(grav)

            if distancia > EPSILON:
                f_x = self.gravedad * (self.posiciones[vertice][0] - grav[0]) / distancia
                f_y = self.gravedad * (self.posiciones[vertice][1] - grav[1]) / distancia

                accum[vertice] = sub_t(accum[vertice], (f_x, f_y))



    def computar_fuerzas_atraccion(self, accum):
        if self.verbose:
            print("Computando fuerza de atraccion...")

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
            fuerza_a = self.calcular_atraccion(distancia)
            f_x = fuerza_a * (self.posiciones[v2][0] - self.posiciones[v1][0]) / distancia
            f_y = fuerza_a * (self.posiciones[v2][1] - self.posiciones[v1][1]) / distancia

            accum[v1] = sum_t(accum[v1], (f_x, f_y))
            accum[v2] = sub_t(accum[v2], (f_x, f_y))

    def computar_fuerzas_repulsion(self, accum):
        if self.verbose:
            print("Computando fuerza de repulsion...")

        # Por cada vertice que esta en el mismo cuadrante calculamos su repulsion
        for v1 in self.grafo[0]:
            for v2 in self.grafo[0]:
                if v1 != v2 and self.mismo_cuadrante(self.posiciones[v1],self.posiciones[v2]):
                    distancia = self.norma(v1,v2)
 
                    # Si estan muy cerca, los alejamos
                    if distancia<EPSILON:
                        num=np.random.rand()
                        self.posiciones[v2] = sum_t(self.posiciones[v2], (num, num))
                        self.posiciones[v1] = sub_t(self.posiciones[v1], (num, num))
                        distancia = self.norma(v1,v2)
                        
                    fuerza_a = self.calcular_repulsion(distancia)
                    f_x = fuerza_a * (self.posiciones[v2][0] - self.posiciones[v1][0]) / distancia
                    f_y = fuerza_a * (self.posiciones[v2][1] - self.posiciones[v1][1]) / distancia
                    accum[v1] = sub_t(accum[v1], (f_x, f_y))
                    accum[v2] = sum_t(accum[v2], (f_x, f_y))

    def actualizar_posiciones(self, vertices, accum):
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
        x1, y1 = self.posiciones[v1]
        x2, y2 = self.posiciones[v2]
        return np.linalg.norm((x1 - x2, y1 - y2))

    def cuadrante(self, vertice):
        mitad_x=MAX_X/2
        mitad_y=MAX_Y/2
        x=vertice[0]
        y=vertice[1]
        cuad=4
        if x<mitad_x and y < mitad_y:
            cuad=1
        elif x>=mitad_x and y < mitad_y:
            cuad=2
        elif x<mitad_x and y >= mitad_y:
            cuad=3
        return cuad    

    def mismo_cuadrante(self,vertice1,vertice2):
        c1=self.cuadrante(vertice1)
        c2=self.cuadrante(vertice2)
        bandera=True
        if c1!=c2:
            bandera=False
        return bandera    


    def calcular_atraccion(self, distancia):
        return (distancia**2)/self.k1

    def calcular_repulsion(self, distancia):
        return (self.k2**2)/distancia

    def init_acumuladores(self, vertices):
        return { vertice: (0, 0) for vertice in vertices }

    def limit(self, n, lim):
        return max(0, min(lim, n))

    def limit_point(self, p):
        return (self.limit(p[0], self.anchura), self.limit(p[1], self.altura))

def lee_grafo_archivo(file_path):
    with open(file_path, "r") as f:
        aristas = []
        cant_v=int(f.readline().rstrip())#rstrip sacatodo los espacios
        vertices = [f.readline().strip("\n") for i in range(cant_v)]

        aristas = [tuple(line.strip().split(' ')) for line in f]
        for v1, v2 in aristas:
            if v1 not in vertices or v2 not in vertices:
                raise Exception("vertice incorrecto")

    return (vertices, aristas)


def main():
    # Definimos los argumentos de linea de comando que aceptamos
    parser = argparse.ArgumentParser()

    # Verbosidad, opcional, False por defecto
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Muestra mas informacion al correr el programa'
    )
    # Cantidad de iteraciones, opcional, 50 por defecto
    parser.add_argument(
        '--iters',
        type=int,
        help='Cantidad de iteraciones a efectuar',
        default=100
    )

    # 
    parser.add_argument(
        '--altura',
        type=int,
        help='Altura de la ventana',
        default=MAX_Y
    )

        # 
    parser.add_argument(
        '--refresh',
        type=int,
        help='refresh',
        default=7
    )

      # 
    parser.add_argument(
        '--anchura',
        type=int,
        help='Anchura de la ventana',
        default=MAX_X
    )
    # Temperatura inicial
    parser.add_argument(
        '--temp',
        type=float,
        help='Temperatura inicial',
        default=10.0
    )
    # Archivo del cual leer el grafo
    parser.add_argument(
        'file_name',
        help='Archivo del cual leer el grafo a dibujar'
    )

    #
    parser.add_argument(
        '--c1',
        type=float,
        help='constante de atraccion',
        default=C_ATRACCION
    )

        #
    parser.add_argument(
        '--c2',
        type=float,
        help='constante de repulsion',
        default=C_REPULSION
    )

            #
    parser.add_argument(
        '--gravedad',
        type=int,
        help='constante de gravedad',
        default=10
    )

                #
    parser.add_argument(
        '--t0',
        type=float,
        help='constante de temperatura',
        default=0.5
    )
    args = parser.parse_args()

    # Descomentar abajo para ver funcionamiento de argparse
   
    # # TODO: Borrar antes de la entrega
    grafo1 = ([1, 2, 3, 4, 5, 6, 7],
              [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1)])
    
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
    
    # # Ejecutamos el layout
    layout_gr.layout()
    return


if __name__ == '__main__':
    main()
