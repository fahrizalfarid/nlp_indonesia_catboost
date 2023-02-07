## run

### requirements

```bash
pip install catboost
```

## main.py

```python
from catboost import CatBoostClassifier
import re


class prediction:
    def __init__(self, modelPath: str):
        self.model = CatBoostClassifier()
        self.model.load_model(modelPath)
        self.label = {}

    def predict(self, text: str):
        def removeSpecialChar(text: str) -> str:
            x = re.sub("[^A-Za-z0-9 ]+", " ", text)
            return x.lower()

        def removeDoubleSpace(text: str) -> str:
            x = re.sub("[ ]{2,}", " ", text)
            return x

        text = removeSpecialChar(text)
        text = removeDoubleSpace(text)

        proba = self.model.predict_proba([text])
        pred = self.model.predict([text])

        self.label["text"] = text
        self.label["negative"] = round(proba[0], 2)
        self.label["neutral"] = round(proba[1], 2)
        self.label["positive"] = round(proba[2], 2)

        print(self.label, pred)


if __name__ == "__main__":
    p = prediction("./sa_100k.ml")
    p.predict("pingin berak")
    p.predict("makanan ini enak sekali")
    p.predict("transfer kelebihan 1000 saja tidak dikasih")
```

## output

```bash
{'text': 'pingin berak', 'negative': 0.48,'neutral': 0.31, 'positive': 0.2} ['negative']
{'text': 'makanan ini enak sekali', 'negative': 0.01,'neutral': 0.11, 'positive': 0.87} ['positive']
{'text': 'transfer kelebihan 1000 saja tidak dikasih','negative': 0.02, 'neutral': 0.91, 'positive': 0.0} ['neutral']
```