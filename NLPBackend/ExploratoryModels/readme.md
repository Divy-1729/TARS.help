# Multi-Label Classification Model Changelog

## 1. EmoAkinator 1
- Initial proof-of-concept model on the [MentalHealth](https://github.com/42Cummer/Catharsis-AI/tree/main/datasets/MentalHealth) dataset.
- Utilized a Bidirectional LSTM without hyperparameter tuning.
- **Accuracy**: 81% (Train), 77% (Validation), 77% (Test)

## 2. EmoAkinator 2
- Implemented an LSTM with fewer units to mitigate overfitting.
- No hyperparameter tuning applied.
- **Accuracy**: 34% (Train), 33% (Validation), 36% (Test)
- **Conclusion**: The [MentalHealth](https://github.com/42Cummer/Catharsis-AI/tree/main/datasets/MentalHealth) dataset exhibits poor quality for this task.

## 3. EmoAkinator 3
- First proof-of-concept model on the [RedditMH](https://github.com/42Cummer/Catharsis-AI/tree/main/datasets/RedditMH) dataset.
- Bidirectional LSTM with 32 units, no hyperparameter tuning.
- **Accuracy**: 88% (Train), 83% (Validation), 83% (Test)
- Shows strong potential, pending hyperparameter optimization.

## 4. EmoAkinator 4
- Basic hyperparameter tuning attempted, limited by computational resources.
- **Accuracy**: 89% (Train), 83% (Validation), 83% (Test)
- Model contains approximately **54 million parameters**.
- Further hyperparameter tuning could push performance to ~85%, but requires more time and compute.

## 5. EmoAkinator 5
- Continued hyperparameter tuning for 2+ hours with minimal gains.
- **Accuracy**: 83.3% (Test)
- Model size remains at **54 million parameters**.
- Realization: an attention mechanism may be necessary for further improvement.

## 6. EmoAkinator 6
- Significant architectural upgrades: added two attention layers, batch normalization, and dropout.
- **Accuracy**: 89% (Train), 83.5% (Validation), 84% (Test)
- Indicates diminishing returns on this dataset without substantial changes; awaiting potential breakthroughs.
