from setuptools import find_packages, setup

setup(
    name="sentiment_service",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "joblib",
        "datasets",
        "fastapi",
        "uvicorn",
    ],
)
