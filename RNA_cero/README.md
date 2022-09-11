
En este archivo se implementa una red neuronal artificial desde cero, para ello se definen algunos conceptos necesarios. 

Se van a definir muchas tecnicas matematicas 


se empieza por definir los parametros: los `pesos` w y los `umbrales` b. 

Los pesos w son los encargados de darle la importancia a cada neurona, estos miden una caracteristica, mientras que el umbral b es un valor constante que mide la facilidad con la que una neurona se enciende.

**Propagacion directa**: 

Se refiere a que dado los parametros, vemos cual es el valor de activacion de salida  $$a^l_j = \sigma(w^l_{jk}\cdot a^{l-1}_k+b^l_j)  =\frac{1}{1+e^{-({w}\cdot{a}+b)}}= \sigma(z^l_j)$$.

**Stochastic Gradient Descent**: 

Es el algoritmo de optimizacion de la red neuronal, basicamente toma un subconjunto de los datos {x} que llamamos `mini-batch`, al conjunto de mini-batches con los que se llegan a tomar todos los datos se llama `epoca`.
Para utilizar este algoritmo es necesario especificar los datos de entrenamiento, el numero de mini-batches que conformaran una epoca y el numero total de epocas que se quieran realizar, tambien se puede dar los datos de prueba para evaluar la eficiencia de la red.

Si se proporciona ``test_data``, la red se evaluará con los datos de prueba después de cada época y se imprimirá el progreso parcial. Esto es útil para acumular progreso, pero ralentiza las cosas sustancialmente.

**Actualizacion de mini-batches**: 

Actualiza los pesos w y umbrales b de la red mediante la aplicación de SGD mediante backpropagation a un solo mini-batch. El `mini_batch` es una lista de tuplas (x, y), `m` es el numero de mini-batches y `eta`, $\eta$ es la tasa de aprendizaje .

$$ w^l_{jk} → w^l_{jk} - \frac{\eta}{m} \frac{\partial C_x}{\partial w^l_{jk}} $$
$$ b^l_j → b^l_j -\frac{\eta}{m}\frac{\partial C_x}{\partial b^l_j} $$


**Backpropagation:**

La idea basica de esto es que nos preguntamos que pasa si hacemos un pequeño cambio a un peso $\Delta w$, esto produce un cambio en la funcion de costo $\Delta C$, esto es en sí, regla de la cadena


$$ \Delta C = \frac{\partial C}{\partial a^L_m}\frac{\partial a^L_m}{\partial a^{L-1}_n}... \frac{\partial a^l_j}{\partial w^l_{jk}}\Delta w^l_{jk} $$

aqui definimos algunas variables 

$z = w \cdot a +b $ 

Activacion:  $\quad \quad a = \sigma{(z)} = \frac{1}{1+e^{-z}}, \quad \quad \sigma'(z) = \sigma(z)(1-\sigma(z))$

Funcion de costo: $\quad \quad C_x = \frac{1}{2} ||y(x) -a^L||^2$

Error: $\quad \quad \delta^L_j = \frac{\partial C}{\partial z^L_j}= \frac{\partial C}{\partial a^L_j}\frac{\partial a^L_j}{\partial z^L_j}= (a^L_j -y(x))\, \sigma'(z^L_j) $

Derivadas parciales de la funcion de costo:
$\frac{\partial C_x}{\partial w^l_{jk}} = \delta^l_j a^{l-1}_k,  \quad \quad \frac{\partial C_x}{\partial b^l_{j}} = \delta^l_j $

En resumen, 


> 1. Input X: propagamos hacia delante una vez y guardamos en memoria las a y z.


> 2. Calculamos el error $\delta^l$ 
$$ \delta^l = \nabla_{a^l}C\cdot\sigma'(z^l) $$



> 3. Propagamos hacia atras el error: 
$$ \delta^l = ((w^{l+1} )^T\delta^{l+1}) \sigma'(z^l)$$



> 4. Salida: Finalmente calculamos las componentes del gradiente $∇_{a^l}C_x$
$$\frac{\partial C_x}{\partial w^l_{jk}} = \delta^l_j a^{l-1}_k,  \quad \quad \frac{\partial C_x}{\partial b^l_{j}} = \delta^l_j $$











