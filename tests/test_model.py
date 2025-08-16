from app.model import vectorizer, model

def test_vectorizer():
    sample_text = ["I love this!"]
    vec = vectorizer.transform(sample_text)
    assert vec.shape[0] == 1

def test_model_prediction():
    sample_text = ["I don't like this!"]
    vec = vectorizer.transform(sample_text)
    pred = model.predict(vec)[0]
    assert pred in [0, 1]
