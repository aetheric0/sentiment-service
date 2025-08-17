import spacy
from sklearn.base import BaseEstimator, TransformerMixin


class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self, model="en_core_web_sm",
        lemmatize=True,
        remove_stopwords=True
    ):
        self.model = model
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        # Load once
        self.nlp = spacy.load(self.model, disable=["parser", "ner"])
        return self

    def transform(self, X):
        docs = self.nlp.pipe(X, batch_size=50)
        cleaned_texts = []
        for doc in docs:
            tokens = []
            for token in doc:
                if self.remove_stopwords and token.is_stop:
                    continue
                if token.is_punct or token.is_space:
                    continue
                text = token.lemma_ if self.lemmatize else token.text
                tokens.append(text.lower())
            cleaned_texts.append(" ".join(tokens))
        return cleaned_texts
