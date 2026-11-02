# LSTM-Based Next Word Prediction

An NLP project that uses a Long Short-Term Memory (LSTM) neural network
to predict the next word in a text sequence. The model is trained on a
quote dataset and made available through a Streamlit web application.

## Live Demo

Try the deployed application here:

**[LSTM-Based Next Word
Prediction](https://lstm-based-next-word-prediction-zxtmunu7amrlkaljuk44qe.streamlit.app/)**

## Project Overview

The application takes a text prompt and predicts the next word using a
trained LSTM model. It can also generate multiple words by repeatedly
adding each prediction to the input and predicting again.

## Features

-   Text preprocessing and tokenization using TensorFlow/Keras
-   N-gram sequence generation for next-word prediction
-   Sequence padding for consistent model inputs
-   Word embeddings and an LSTM neural network
-   Next-word prediction from model output probabilities
-   Interactive Streamlit application
-   Saved model and tokenizer files for inference without retraining

## Technologies Used

-   Python
-   TensorFlow / Keras
-   NumPy
-   Streamlit
-   Jupyter Notebook
-   Google Colab

## Repository Structure

``` text
.
├── app.py                    # Streamlit application
├── LSTM_implementation.ipynb # Model implementation and training notebook
├── lstm_model.h5             # Trained LSTM model
├── max_len.pkl               # Saved maximum input sequence length
├── qoute_dataset.csv         # Quote dataset used for training
├── requirements.txt          # Python dependencies
├── tokenizer.pkl             # Fitted tokenizer
└── README.md
```

> Windows may hide file extensions in File Explorer. The names above
> assume `app` is `app.py`, `LSTM_implementation` is
> `LSTM_implementation.ipynb`, `qoute_dataset` is `qoute_dataset.csv`,
> and `requirements` is `requirements.txt`. Keep the actual filenames
> consistent with the paths used in the code.

## How It Works

1.  **Data preparation:** The quote dataset is loaded and its text is
    prepared for training.
2.  **Tokenization:** Keras `Tokenizer` converts words into integer IDs.
3.  **Sequence generation:** N-gram sequences are created so the
    preceding words can be used to predict the next word.
4.  **Padding:** Sequences are padded to a consistent length before
    being passed to the model.
5.  **Model training:** An Embedding layer learns word representations,
    an LSTM layer learns sequential patterns, and a Dense softmax layer
    produces output scores.
6.  **Prediction:** The input text is tokenized and padded. The model
    predicts an output index, which is mapped back to a word using the
    saved tokenizer.
7.  **Text generation:** To generate multiple words, each predicted word
    is appended to the input and the prediction process is repeated.

## Setup and Installation

Python 3.11 is a suitable choice for running the Streamlit application.

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment (Windows)

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

## Run the Streamlit Application

Make sure these files are in the project folder:

-   `app.py`
-   `lstm_model.h5`
-   `max_len.pkl`
-   `tokenizer.pkl`

Run:

``` bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually
`http://localhost:8501`. Open that address in your browser.

## Model Training

The `LSTM_implementation.ipynb` notebook contains the model
implementation and training workflow. Open it in Google Colab or Jupyter
Notebook and run its cells in order.

The training notebook and Streamlit application must use the **same
model, tokenizer, and sequence-length configuration**. The
`lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` files should all
come from the same training run. Do not fit a new tokenizer for
inference.

The `qoute_dataset.csv` file is the dataset used by the project. The
application uses the saved model assets for prediction and does not need
to retrain the model every time it starts.

## Saved Model Files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `lstm_model.h5`                     Trained LSTM model

  `tokenizer.pkl`                     Fitted tokenizer containing the
                                      word-to-index mapping

  `max_len.pkl`                       Sequence length used to prepare
                                      model inputs
  -----------------------------------------------------------------------

## Limitations

-   Predictions depend on patterns learned from the quote dataset and
    may not always be grammatically or contextually appropriate.
-   The model can predict only words represented in its vocabulary.
-   The tokenizer and model must correspond to the same training run.
-   No accuracy score is claimed in this README; add evaluation metrics
    only after measuring performance on held-out data.

## Future Improvements

-   Evaluate the model on a held-out test set.
-   Report Top-1 and Top-5 prediction accuracy.
-   Experiment with sequence length, embedding dimensions, and LSTM
    units.
-   Improve input validation and user feedback.
-   Compare the LSTM with simpler RNN and baseline language models.

## Author

**\[Parth Gupta\]**

