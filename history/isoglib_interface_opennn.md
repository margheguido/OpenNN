- Neuralnetwork contiente un oggetto  OutputFunction che a sua volta ne contiente uno IsoglibInterface

- Il constructor di IsoglibInterface prende in ingresso:

  * nome (con percorso) della directory in cui c'è la mesh

  * nome del file binario da cui leggere la soluzione

  Quindi anche il constructor di NeuralNetwork e di OutputFunction prendono in ingresso queste due CloseElement

- Nel corpo del constructor di IsoglibInterface viene       eseguita la funzione set_problem_resolution, che legge il file della mesh e crea i due memebri Problem e TimeAdvancing

- Nella classe IsoglibInterface è presente anche il membro nDof, che specifica in quanti nodi è calcolata la soluzione (serve a sapere quando fermarsi nella lettura del file binario).
Bisogna capire quando settarlo.
