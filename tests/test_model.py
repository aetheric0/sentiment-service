from app.model import model_pipeline


def test_model_pipeline_loaded():
    assert model_pipeline is not None
    assert hasattr(model_pipeline, "predict")


def test_pipeline_predicts():
    text = "This is fantastic!"
    pred = model_pipeline.predict([text])
    assert pred in [0, 1]


def test_inference_speed():
    import time
    text = "I love this movie!"
    start = time.time()
    _ = model_pipeline.predict([text])
    duration = time.time() - start
    assert duration < 0.5  # seconds
