#! /usr/bin/python

# 6ta Practica Laboratorio 
# Complementos Matematicos I
# Ejemplo parseo argumentos

import argparse
from re import S
from termios import VT1
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

class LayoutGraph:

    def __init__(self, grafo, iters, refresh, c1, c2, altura,anchura ,verbose=False):
        """
        Parámetros:
        grafo: grafo en formato lista
        iters: cantidad de iteraciones a realizar
        refresh: cada cuántas iteraciones graficar. Si su valor es cero, entonces debe graficarse solo al final.
        c1: constante de repulsión
        c2: constante de atracción
        verbose: si está encendido, activa los comentarios
        """

        # Guardo el grafo
        self.grafo = grafo

        # Inicializo estado
        # Completar
        self.posiciones = {}
        self.fuerzas = {}

        # Guardo opciones
        self.iters = iters
        self.verbose = verbose
        self.altura = altura
        self.anchura = anchura
        self.gravedad = 2
        # TODO: faltan opciones
        self.refresh = refresh

        # ver si hace falta
        self.c1 = c1
        self.c2 = c2

        self.k1 = c1*np.sqrt((anchura*altura)/len(grafo[0]))
        self.k2 = c2*np.sqrt((anchura*altura)/len(grafo[0]))

    def init_posiciones(self, vertices):
        return {
            key: tuple(np.random.randint(101,size=2))
            for key in vertices
        }

    def dibujar_grafo(self):
        plt.clf()
        for v1 , v2 in self.grafo[1]:

            plt.plot(
                (self.posiciones[v1][0], self.posiciones[v2][0]),
                (self.posiciones[v1][1], self.posiciones[v2][1]),
                'g',
                marker='o',
                mfc='black'
            )
        for v in self.grafo[0]:
            plt.annotate(v, xy=(self.posiciones[v][0], self.posiciones[v][1] + 5))
        plt.pause(1)


    def layout(self):
        """
        Aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
        un layout
        """
        const=0
        self.posiciones = self.init_posiciones(self.grafo[0])
        for it in range(0,self.iters):
            if const*self.refresh == it:
                self.dibujar()
                const+=1
        ##print("aca hicimos la iteracion ",it)
            self.step()

        plt.show()


    def step(self):
        accum_x, accum_y = self.init_acumuladores(self.grafo[0])
        self.computar_fuerzas_atraccion(accum_x,accum_y)
        self.computar_fuerzas_repulsion(accum_x,accum_y)
        self.computar_gravedad(self.grafo[0], accum_x,accum_y)
        self.actualizar_posiciones(self.grafo[0],accum_x, accum_y)
      
    def computar_gravedad(self, vertices, accum_x, accum_y):
        # for vertice in vertices:
        #     x, y = self.posiciones[vertice]
        #     grav_x = self.anchura / 2 - x
        #     grav_y = self.altura / 2 - y
            # norma = np.sqrt(grav_x ** 2 + grav_y ** 2)
        #     grav = (grav_x / norma * self.gravedad, grav_y / norma * self.gravedad)

        #     accum_x[vertice] += grav[0]
        #     accum_y[vertice] += grav[1]

        for vertice in self.grafo[0]:
            x, y = self.posiciones[vertice]
            grav_x = self.anchura / 2 - x
            grav_y = self.altura / 2 - y
            distancia = np.sqrt(grav_x ** 2 + grav_y ** 2)

            if distancia:
                f_x = self.gravedad * (self.posiciones[vertice][0] - grav_x) / distancia
                f_y = self.gravedad * (self.posiciones[vertice][1] - grav_y) / distancia
                
                accum_x[vertice] += f_x
                accum_y[vertice] += f_y



    def computar_fuerzas_atraccion(self, accum_x, accum_y):
        for v1 , v2 in self.grafo[1]:
            distancia = self.norma(v1,v2)
            if distancia:
                fuerza_a = self.calcular_atraccion(distancia)
                f_x = fuerza_a * (self.posiciones[v1][0] - self.posiciones[v2][0]) / distancia
                f_y = fuerza_a * (self.posiciones[v1][1] - self.posiciones[v2][1]) / distancia

                accum_x[v1] += f_x
                accum_y[v1] += f_y
                accum_x[v2] -= f_x
                accum_y[v2] -= f_y

    def computar_fuerzas_repulsion(self, accum_x, accum_y):
        for v1 in self.grafo[0]:
            for v2 in self.grafo[0]:
                if v1 != v2:
                    distancia = self.norma(v1,v2)
                    if distancia:
                        fuerza_a = self.calcular_repulsion(distancia)
                        f_x = fuerza_a * (self.posiciones[v1][0] - self.posiciones[v2][0]) / distancia
                        f_y = fuerza_a * (self.posiciones[v1][1] - self.posiciones[v2][1]) / distancia
                        accum_x[v1] -= f_x # Invertimos esto ultimo
                        accum_y[v1] -= f_y
                        accum_x[v2] += f_x
                        accum_y[v2] += f_y

    def actualizar_posiciones(self, vertices, accum_x, accum_y):
        for vertice in vertices:
            x = self.posiciones[vertice][0] + accum_x[vertice]
            y = self.posiciones[vertice][1] + accum_y[vertice]
            if x > 100:
                x = 100
            if x < 0:
                x = 0
            if y > 100:
                y = 100
            if y < 0:
                y = 0
            self.posiciones[vertice] = (x, y)
            # self.posiciones[vertice][1] += accum_y[vertice]

    def norma(self, v1, v2):
        # x1, y1 = self.posiciones[v1]
        # x2, y2 = self.posiciones[v2]
        # return np.linalg.norm((x1 - x2, y1 - y2))
        x1, y1 = self.posiciones[v1]
        x2, y2 = self.posiciones[v2]
        return np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)


    def calcular_atraccion(self, distancia):
        return (distancia**2)/self.k1

    def calcular_repulsion(self, distancia):
        return (self.k2**2)/distancia

    def init_acumuladores(self, vertices):
        ## ver de fusionar los acum
        accum_x = {}
        accum_y = {}

        for vertice in vertices:
            accum_x[vertice] = 0
            accum_y[vertice] = 0

        return [accum_x, accum_y]

    def dibujar(self):
        self.dibujar_grafo()
    

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
        default=50
    )

    # 
    parser.add_argument(
        '--altura',
        type=int,
        help='Altura de la ventana',
        default=100
    )

      # 
    parser.add_argument(
        '--anchura',
        type=int,
        help='Anchura de la ventana',
        default=100
    )
    # Temperatura inicial
    parser.add_argument(
        '--temp',
        type=float,
        help='Temperatura inicial',
        default=100.0
    )
    # Archivo del cual leer el grafo
    parser.add_argument(
        'file_name',
        help='Archivo del cual leer el grafo a dibujar'
    )

    args = parser.parse_args()

    # Descomentar abajo para ver funcionamiento de argparse
    #print(args.verbose)
    #print(args.iters)
    #print(args.file_name)
    #print(args.temp)
    #return

    # # TODO: Borrar antes de la entrega
    grafo1 = ([1, 2, 3, 4, 5, 6, 7],
              [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1)])
    
    # # Creamos nuestro objeto LayoutGraph
    layout_gr = LayoutGraph(
         grafo = lee_grafo_archivo(args.file_name),
         iters=args.iters,
         refresh=1,
         c1=8,
         c2=0.6,
         altura=args.altura,
         anchura=args.anchura,
         verbose=args.verbose
     )
    
    # # Ejecutamos el layout
    layout_gr.layout()
    return


if __name__ == '__main__':
    main()
    # try:
    #     lee_grafo_archivo('Cubo.txt')
    # except Exception as e:
    #     print('Ocurrio una excepcion', e)
    # main()
