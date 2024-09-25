from fastapi import FastAPI
from xaiensembleapi import xaiensemble as e_xai
import tensorflow as tf
from keras.models import load_model
from typing import Dict, Any
from fastapi.responses import HTMLResponse

import numpy as np
import os
class MyAPI:
    def __init__(self):
        # Inicializa o FastAPI
        self.app = FastAPI()
        self.base_dir = os.path.dirname(__file__)
        self.features_name = ["TGS-826","TGS-2611","TGS-2603","TGS-813","TGS-822","TGS-2602","TGS-823"]
        self.class_names = ['Albicans', 'Glabrata', 'Haemulonii', 'Kodamaea_ohmeri', 'Krusei', 'Parapsilosis']
        self.main, self.api, self.X_train, self.y_train,self.X_test, self.y_test, self.SAMPLE_INDEX, self.model = self.load_data_and_model(self.features_name, 5)
        # print("X_test: "+str(self.X_test))
        # Define os dados
        self.dataFrame = {
            0: {0: 1000, 1: 2200, 2: 00000, 3: 3000, 4: 35555, 5: 4859},
            1: {0: 1000, 1: 9333, 2: 00000, 3: 500, 4: 35555, 5: 4859},
            2: {0: 1000, 1: 500, 2: 00000, 4: 3000, 4: 600, 5: 4859}
        }

        # Define as rotas
        self.create_routes()

    def create_routes(self):    
        # Define rota GET "/"
        @self.app.get("/", response_class=HTMLResponse)
        def home():
            html_content = """
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>API XAI Ensemble</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background-color: #f4f4f4;
                    }
                    .container {
                        text-align: center;
                        background: #fff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }
                    h1 {
                        color: #333;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>API XAI Ensemble to VOCs identification</h1>
                    <h4>Use /docs at the end of the URL to see all methods</h4>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)

        # Define rota POST "/prediction/"
        @self.app.post("/prediction/")
        def get_prediction(input: Dict[str, Any]):
            input = np.array(list(input.values()))
            result = self.api.get_predicted_class(input[0], self.model)
            return {"Result":str(result)}

        # Define rota GET "/resultAI/"
        @self.app.post("/mostImportantFeatures/")
        def get_result(input: Dict[str, Any]):
            # result = 10
            input = np.array(list(input.values()))
            # dicionario = {i: valor for i, valor in enumerate(self.X_test[self.SAMPLE_INDEX])}
            df_LIME, df_shap, df_GRAD, df_final, df_caounts = self.main.run_2_methods_mult(self.X_train, input[0], self.model, self.class_names,'conv1d_30',4)
            text_result, main_fungi, main_sensors =  self.main.get_result_summary(df_caounts,self.base_dir+'/src/MappingVOC_DB.db',self.api.QUERY_FUNGI, self.api.QUERY_SENSOR)
            return {"text_result":text_result,"main_sensors":main_sensors,"main_fungi":main_fungi}
        @self.app.post("/sampleFromDB/")
        def get_semplesFromDB(input: Dict[str, Any]):
            # result = 10
            input = np.array(list(input.values()))
            data = self.api.get_samples_from_db(self.X_train, self.y_train, self.api.get_predicted_class(input[0], self.model))
            dicionario_dfs = {f'sample{i+1}': df.to_dict(orient='records') for i, df in enumerate(data)}
            print(dicionario_dfs)
            return dicionario_dfs
    def load_data_and_model(self, features_name, SAMPLE_INDEX):
        # features_name = ["TGS-826","TGS-2611","TGS-2603","TGS-813","TGS-822","TGS-2602","TGS-823"]
        # class_names = ['Albicans', 'Glabrata', 'Haemulonii', 'Kodamaea_ohmeri', 'Krusei', 'Parapsilosis']
        api = e_xai.ExplainableAPI(features_name)
        
        X_train, y_train = api.load_data(path=self.base_dir+"/src/AllCandidas_TRAIN.csv", sep=",")  # Replace with actual path and separator
        X_test, y_test = api.load_data(path=self.base_dir+"/src/AllCandidas_TEST.csv", sep=",")  # Replace with actual path and separator
        # SAMPLE_INDEX = 8# amostra que será selecionada do df de teste
        model = tf.keras.models.load_model(self.base_dir+'/src/best_model.hdf5')

        main = e_xai.Run_methods(api)
        print("Works well")
        return main, api, X_train, y_train, X_test, y_test, SAMPLE_INDEX, model
# Cria uma instância da classe MyAPI
api_instance = MyAPI()
# api_instance.get_predicted_class(X_test[SAMPLE_INDEX], model)

app = api_instance.app
