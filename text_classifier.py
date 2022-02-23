import ktrain
def load_predictor(model_path):
    return ktrain.load_predictor(model_path)

def text_to_emotion(text):
    predictor = load_predictor('./predictor')
    labels = ['happiness', 'sadness', 'fear', 'anger', 'neutral']
    predictions = predictor.predict(text,return_proba=True)
    return dict(zip(labels,predictions))
