1) Confrontare il main con quello su git e recuperare la parte di scaling.
	Ho inserito nel main su gti la parte di scaling che c'era nel tuo main, pare 		funzionare anche con due output
2) Cosa ho modificato nei file di opennn? Cos'� testing_analysis?
	nei file di opennn non c'� praticamente niente di modificato(sono solo spazi o 		cose del genere) eccetto in testing analysis dove avevi aggiunto dei print.
	Testing analysis � l'istanza che si crea per testare la rete. Sto cercando ora di 	capire come fare a farla funzionare con pi� output. infatti fino ad ora abbiamo 	inserito lo "sdoppiamento" dell'output fuori dalle operazioni che esegue la rete 	quando calcola l'output (dato che non abbiamo inserito nuovi parametri), lo 		abbiamo inglobato nel calcolo della loss e del gradiente.

	Ha senso inserire la funzione che calcola l'output multidimensionale a partire da 	quello della rete anche in Testing_analysis? (E' gi� presente in normalized 		squared error) 
	oppure dovremmo creare una classe che fa questo e le altre operazioni associate 	tipo calcolare il gradiente?

Ho usato Gradient Descent come training method, a grandi linee i punti in cui ho inserito qualcosa gli ho elencati nel file GradientDescent2outputs


Non ho capito come funzionano scaling e unsealing layer. Non facciamo lo scaling usando scale_inputs/targets_minimum_maximum?

PROBLEMA: ho due targets, quando fa lo scaling lo fa in modo indipendente per i due:
calcola min e max del primo target e scala e poi idem con il secondo
A noi invece serve che abbiano lo stesso range, dato che calcoleremo il secondo output in funzione del primo 
inserisco nella funzione scale_columns_minimum_maximum di matrix.h (che � quella in cui effettivamente viene fatto lo scaling) una variabile bool che di default � 0, che viene settata a uno se vogliamo fare lo scaling in modo uniforme 
conseguentemente deve essere prima passata alla funzione di data set  scale_targets_minimum_maximum -> Adesso sembra funzionare lo scaling dei target ma la loss � altissima: c'� qualcosa che non va
