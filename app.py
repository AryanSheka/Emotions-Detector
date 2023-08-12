from flask import Flask, request, render_template
from src.Pipelines.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sentiment', methods = ['POST','GET'])
def predict_sentiment():
    if(request.method == 'GET'):
        return render_template('home.html')
    
    else:
        Sentence = request.form.get('sentence')
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(Sentence)
        return render_template('home.html', answer = result)
    
if __name__ == '__main__':
    app.run(debug = True)