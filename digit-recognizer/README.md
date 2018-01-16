[[Keras] Digit Recognizer competition on Kaggle using Neural Network](https://qiita.com/SHUAI/items/25b7eb1919e944534a90)

### Keras version: 2.0.5
default epochs in the fit method of the Sequential model is 10  
https://github.com/keras-team/keras/blob/2.0.5/keras/models.py#L800

#### functional way
set epochs=10 in the fit method  
https://github.com/Jim-Lin/kaggle-competition/blob/master/digit-recognizer/run_functional.py#L22
