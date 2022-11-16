from cgitb import text
from flask import Flask, render_template, request, send_file
from waitress import serve
import numpy as np
import pickle
from ModelClassifier import text_processing

filename = 'C:/Users/eduar/Documents/Processing/libraries/News_Classifier/ModelClassifier.sav'
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():   
    return render_template('/index.html')
    
@app.route("/busqueda", methods = ['POST'])
def predict():    
    if request.method == 'POST':        
        busqueda = request.values['noticia']  
        print("3#######################################################################")  
        print("3#######################################################################")  
        print("3#######################################################################")  
        print(busqueda)
        print("3#######################################################################")  
        vector = text_processing(busqueda)

        data = []
        for val in range(vector.shape[0]):            
            data.append(vector[val])
        print(data)

        noticias_ = ""
        noticias_ += "<div><p><a href='/predict?v1="+str(data[0])
        for val in range(1, vector.shape[0]):
            noticias_ +="&v"+str(val+1)+"="+str(data[val])
        noticias_ +="'>Resultado</a></p></div>"
        print(noticias_)
        
        return render_template('/noticias.html', noticias = noticias_)
    else:
        return "ERROR"
    
@app.route("/predict", methods = ['GET'])
def predict2():    
    if request.method == 'GET': 

        ################################################################
        ########### Reconstruyendo Peticion ############################
        ################################################################
        vector = []
        for r in request.args:
            #print(r)            
            vector.append(request.values[r])                
        vector = np.asarray(vector, dtype=np.float32)
        #print(vector)
        vector = np.expand_dims(vector, 0)
        print(vector.shape)
        #TODO HACER QUE EL MODELO PUEDA RECIBIR LOS 1000 ELEMENTOS DEL VECTOR // COMO PASARLE EL VECTOR
        resultado = loaded_model.predict(vector)
        if(resultado == 0):
            resultado = "Noticia Buena!"  
        else:
            resultado = "Noticia Mala!"                           
        return render_template('/prediccion.html', resultado = resultado)      
    else:
        return "ERROR"

if __name__ == "__main__":            
    print("corriendo")
    serve(app, host="0.0.0.0", port=5000)