

# Multi Label Classification Model Changelog

## 1. EmoAkinator1
- Proof of Concept Model on the [MentalHealth](https://github.com/42Cummer/Catharsis-AI/tree/main/datasets/MentalHealth) dataset
- Bidirectional LSTM with no hyperparameter tuning
- Accuracy : 81% train, 77% val, 77% test
## 2. EmoAkinator 2
- No hyperparameter tuning, LSTM with lesser units to combat overfitting
- Accuracy : 34% train, 33% val, 36% test
- Conclusion : [MentalHealth](https://github.com/42Cummer/Catharsis-AI/tree/main/datasets/MentalHealth) dataset is ass.

## 3. EmoAkinator 3
- First Proof of Concept model on the [RedditMH](https://github.com/42Cummer/Catharsis-AI/tree/main/datasets/RedditMH) dataset
- Bidirectional LSTM with 32 units, no hyperparameter tuning
- Accuracy : 88% test, 83% val, 83% test
- Has potential, but needs hyperparameter tuning.

## 4. EmoAkinator 4
- Tried some hyperparameter tuning, but it's so slow
- Accuracy : 89% test, 83% val, 83% test (gotta love the 0.3% increase ngl)
- Also 54 Million parameters what the hell
- Some good hyperparameter tuning can take this to 85%, but I need more resources and time!!!

  ## 5. EmoAkinator 5
- Hyperparameter tuning for 2 hours, still so slow 
- Accuracy : 83.3% test (gotta love the 0.3% increase ngl)
- Also 54 Million parameters what the hell
- I was wrong, I should add an attention layer

## 6. EmoAkinator 6
- Major improvements from EmoAkinator 5
- Added two attention layers, batch_normalization and dropout
- Accuracy : 89% train, 83.5 val, 84% test
- I do not see a lot of improvement on this dataset from this, unless **Max** cooks hard.
