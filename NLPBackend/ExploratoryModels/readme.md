# Binary Classification Model Changelog

## Basic Model
1. **BasicModel9600**: A very basic model to test feasbility on the `MentalHealth` dataset. Achieved $99\%$ accuracy on train dataset, $78\%$ accuracy on validation dataset and $78\%$ accuracy on test dataset.

2. **BasicModel9650**: Another very basic model to test feasbility on the `MentalHealth` dataset, no hyperparameter tuning slightly-modified version of **BasicModel9600**. Achieved $99.64\%$ accuracy on train dataset, $80\%$ accuracy on validation dataset and $78.18\%$ accuracy on test dataset, so slight improvements.

## FNet Model
3. **FentTweaker9700**: Just testing FNet on `SuicideWatch` dataset. Proof-of-concept model.
   
4. **FentTweaker9705**: Simple binary classifier model on the `SuicideWatch` dataset, built from **FentTweaker9699**. Model uses `FNetEncoder` layers to speed up training and handle long input sequences easily. Achieved $99%$ accuracy on train dataset, $65%$ accuracy on validation dataset and $65%$ accuracy on the test dataset. *Note: model is overfitting and hyperparams not tuned.*

5. **FentTweaker9710**: Upgraded version of previous model with hyperparameter tuning. However, performance is still shitty. Trying `objective = "val_loss"` instead of `objective = "val_accuracy"` for the next version. Model is still overfitting after the first few epochs.

6. **FentTweaker9715**: Further hyperparameter tuning on the previous version. Achieved $90$ accuracy on train dataset, $68$ accuracy on validation dataset, and $69$ accuracy on test dataset. Gotta love that $4$ improvement in accuracy (at least progress is being made). Model is *still* overfitting on the first few epochs

7. **FentTweaker9720**: Major changes to **FentTweaker9715** as my fucking dumbass used `test_size=0.9` (10/90 train/test) instead of `test_size=0.1`  (90/10 train/test) which is obviously wrong. Also added `ReduceLROnPlateau` callback and changed `EarlyStopping` callback parameters for the benefit of the model. Model is still overfitting but in the later stages of training. Perhaps try a smaller lr and different patience level and/or other params for the callbacks. Current model has $81$ accuracy on train dataset, $73$ accuracy on validation dataset and $71$ accuracy on test dataset.

8. **FentTweaker9725**: Minor updates to **FentTweaker9720**, dropout rate `0.1 -> 0.15`. Experimented around with `ReduceLROnPlateau` callback and `EarlyStopping` callback parameters for the benefit of the model. Fluctuations in `val_loss` are normal, but the plateau whilst training loss decreases (around epoch 11) indicates model is at capacity and beginning to overfit on train dataset. Try using a learning rate scheduler instead of ReduceLROnPlateau callback. Maybe also try adding another `FNetEncoder` layer and increasing dropout to `0.20`. Current model has $85$ accuracy on train dataset, $73$ accuracy on validation dataset and $73$ accuracy on test dataset.

9. **FentTweaker9730**: Minor updates to **FentTweaker9725**, tried using a single FNetLayer (corrrespondingly decreased dropout to `0.1`); also slight tweaks to `ReduceLROnPlateau` and `EarlyStopping`. Model is still overfitting (around epoch 5), but train accuracy has increased to $85$ and validation accuracy to $75$. However, test accuracy drops to $59$.

10. **FentTweaker9740**: Major updates to **FentTweaker9730**. Tried two FNet layers with a dropout layer in between (`rate = 0.10`) to reduce number of parameters and hence overfitting. Added `ModelCheckpoint` callback to save model parameters since `EarlyStopping` doesn't actually save the best model. Train accuracy has increased to $87$, but validation accuracy decreases to $83$. However, test accuracy drops to $69$.
  
12. 9799: ~~Also used FNetTokenizer as done in the official implementation except there's an error that I'm strategically not solving until we get accepted.~~

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
- Tried some hyperparameter tuning, but it's so fucking slow I gave up
- Accuracy : 89% test, 83% val, 83% test (gotta love the 0.3% increase ngl)
- Also 54 Million parameters what the hell
- Some good hyperparameter tuning can take this to 85%, but I need more resources and time!!!

  ## 5. EmoAkinator 5
- Hyperparameter tuning for 2 hours, still so fucking slow I gave up
- Accuracy : 83.3% test (gotta love the 0.3% increase ngl)
- Also 54 Million parameters what the hell
- I was wrong, I should add an attention layer

## 6. EmoAkinator 6
- Major improvements from EmoAkinator 5
- Added two attention layers, batch_normalization and dropout
- Accuracy : 89% train, 83.5 val, 84% test
- I do not see a lot of improvement on this dataset from this, unless **Max** cooks hard.
