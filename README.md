# EMI2024
Escuela de Verano en Métodos Iterativos 2024 @ Centro de Modelamiento Matemático (CMM), av. Beauchef 851 piso 7, sala von Neumann.

## Organizadores
- Nicolás A. Barnafi
- Manuel A. Sanchez


## Programa
Día 1 – Jueves 4

| Hora          | Actividad |
| ------------- | --------- |
| 09:30 – 11:00 | Bienvenida + Introducción |
| 11:00 – 11:30 | Coffee break |
| 11:30 – 13:00 | Métodos Iterativos: Richardson, Jacobi, GaussSeidel, SOR, Matriz de iteracion - analisis de convergencia |
| 13:00 – 14:30 | Libre |
| 14:30 – 16:00 | Métodos de Krylov: CG y GMRES |
| 16:00 – 16:30 | Coffee break |
| 16:30 – 18:00 | Métodos de Krylov: Cálculo de valores propios |

Día 2 – Viernes 5

| Hora          | Actividad |
| ------------- | --------- |
| 09:30 – 11:00 |Métodos de Krylov: Precondicionadores |
| 11:00 – 11:30 |Coffee break |
| 11:30 – 13:00 |Precondicionadores optimales: Multigrid y domain decomposition |
| 13:00 – 14:30 |Libre |
| 14:30 – 16:00 |Problemas avanzados |
| 16:00 – 16:30 |Coffee break |
| 16:30 – 18:00 |Trabajo libre |


## Instalación

Con los siguientes comandos se instala NGSolve, Jupyter (para abrir notebooks) y el backend de WebGL para visualizar localmente. 

    pip install ngsolve
    pip install jupyter
    pip install webgui_jupyter_widgets

También se puede usar Google Colab siguiendo las instrucciones disponibles en [Fem on Colab](https://fem-on-colab.github.io/packages.html). Recomendamos compilar la librería siguiendo las instrucciones de la documentación para agregar soporte para MPI, UMFPACK y MUMPS, disponible [acá](https://docu.ngsolve.org/latest/install/installlinux.html).

*Ojo con la compilación local, ya que el link a la carpeta python lo tienen escrito para python3, y en Ubuntu 22.04 la carpeta en realidad se llama python3.10.*

## Referencias

- Y Saad. Iterative methods for sparse linear systems. Society for Industrial and Applied Mathematics, 2003.
- AJ Wathen. Preconditioning. Acta Numerica, 2015.

Gran parte del material del curso está basado en el libro de Saad, además de material que teníamos generado desde antes. La segunda referencia es una Acta Numerica, que es una familia de papers que se hacen _a pedido_, o sea que ese tal Wathen sabía *tanto* de precondicionadores, que le pidieron que haga un compendio del estado del arte. Es una lectura bastante relajada, clarísima y muy vigente.
