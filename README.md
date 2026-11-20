# Sentence Completion and Next-Word Prediction

An NLP project exploring next-word prediction and sentence completion
using two approaches:

1.  **LSTM (trained model):** A word-level LSTM model trained on a quote
    dataset and deployed through a Streamlit application.
2.  **GPT-2 (pretrained model experiment):** A small experiment using
    the pretrained GPT-2 language model from Hugging Face Transformers
    to inspect tokenization, model logits, and predict the next token.

The LSTM implementation is the main end-to-end project. The GPT-2
notebook is a separate learning experiment and is not fine-tuned on the
quote dataset.

## Live Demo

Try the deployed LSTM application:

**[LSTM Next-Word
Predictor](https://lstm-based-next-word-prediction-zxtmunu7amrlkaljuk44qe.streamlit.app/)**

## Project Overview

The project explores how language models use preceding text to predict
what comes next.

-   The **LSTM application** accepts a text prompt and predicts the next
    word. It can generate multiple words by repeatedly appending each
    prediction to the input and running the model again.
-   The **GPT-2 notebook** loads a pretrained GPT-2 model, tokenizes a
    prompt, examines the output logits, and selects the most likely next
    token using `argmax`.

These are two different approaches: the LSTM is trained as part of this
project, while GPT-2 uses pretrained weights.

## Features

### LSTM implementation

-   Text preprocessing and tokenization using TensorFlow/Keras
-   N-gram sequence generation for next-word prediction
-   Sequence padding for consistent model inputs
-   Embedding layer and LSTM neural network
-   Next-word prediction from model output probabilities
-   Iterative multi-word generation
-   Streamlit interface for interactive inference
-   Saved model and tokenizer assets, so the app does not need to
    retrain at startup

### GPT-2 experiment

-   Tokenization with `GPT2Tokenizer`
-   Loading a pretrained `GPT2LMHeadModel`
-   Inspection of model output logits
-   Extraction of the logits at the final input position
-   Greedy next-token selection using `torch.argmax`
-   Decoding the predicted token back into text

> **Scope note:** The GPT-2 notebook currently demonstrates next-token
> inference for a prompt. It does not fine-tune GPT-2, train a language
> model, or implement a complete text-generation loop.

## Technologies Used

-   Python
-   TensorFlow / Keras
-   PyTorch
-   Hugging Face Transformers
-   NumPy
-   Streamlit
-   Jupyter Notebook
-   Google Colab

## Repository Structure

The layout below keeps the deployed LSTM project at the repository root
and groups the GPT-2 experiment in its own folder.

``` text
.
├── app.py                              # Streamlit application for the LSTM model
├── LSTM_implementation.ipynb           # LSTM implementation and training notebook
├── lstm_model.h5                       # Trained LSTM model
├── tokenizer.pkl                       # Fitted Keras tokenizer
├── max_len.pkl                         # Saved input sequence length
├── qoute_dataset.csv                   # Quote dataset used for LSTM training
├── GPT2_Next_Token_Prediction.ipynb    # Pretrained GPT-2 inference experiment
├── requirements.txt                    # Dependencies for the Streamlit/LSTM app
└── README.md
```

Keep the filenames and folder locations consistent with the paths used
in your code. If your actual dataset filename is spelled
`quote_dataset.csv` rather than `qoute_dataset.csv`, update the name in
this README to match the file in your repository.

## How the LSTM Model Works

1.  **Data preparation:** The quote dataset is loaded and its text is
    prepared for training.
2.  **Tokenization:** Keras `Tokenizer` converts words into integer IDs.
3.  **Sequence generation:** N-gram sequences are created so preceding
    words can be used to predict the next word.
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

## How the GPT-2 Experiment Works

1.  **Tokenization:** `GPT2Tokenizer` converts the input prompt into
    token IDs and creates PyTorch tensors.
2.  **Model loading:** `GPT2LMHeadModel.from_pretrained("gpt2")` loads
    the pretrained GPT-2 model.
3.  **Forward pass:** The tokenized prompt is passed to the model.
4.  **Logits:** The model returns logits for each input-token position
    and every token in the GPT-2 vocabulary.
5.  **Last-position selection:** The logits at the final input position
    are selected because they represent the model's next-token
    predictions.
6.  **Greedy prediction:** `torch.argmax` selects the token with the
    highest logit.
7.  **Decoding:** The selected token ID is decoded into readable text.

This notebook is intended to demonstrate the basic next-token prediction
process. It is separate from the LSTM model and does not use the LSTM
tokenizer or its trained weights.

## Setup and Installation

Python 3.11 is a suitable starting point for the Streamlit application.
Install the dependencies in a virtual environment.

### 1. Clone the repository

``` bash
git clone <https://github.com/parthgupta5555/LSTM-Based-Next-Word-Prediction>
cd <LSTM-Based-Next-Word-Prediction>
```

Replace the placeholders with your actual GitHub repository URL and
folder name.

### 2. Create and activate a virtual environment (Windows)

``` bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install the LSTM application dependencies

``` bash
pip install -r requirements.txt
```

### 4. Install GPT-2 experiment dependencies

If you want to run the GPT-2 notebook, install PyTorch and Hugging Face
Transformers as well:

``` bash
pip install torch transformers
```

Choose a PyTorch installation command appropriate for your operating
system and hardware from the [official PyTorch installation
page](https://pytorch.org/get-started/locally/). The GPT-2 notebook
downloads pretrained model files the first time it runs, so an internet
connection is required for the initial download.

## Run the Streamlit Application

Make sure these files are in the repository root, alongside `app.py`:

-   `app.py`
-   `lstm_model.h5`
-   `tokenizer.pkl`
-   `max_len.pkl`

Run:

``` bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually
`http://localhost:8501`. Open that address in your browser.

## Model Training and Saved Assets

The `LSTM_implementation.ipynb` notebook contains the LSTM
implementation and training workflow. Open it in Google Colab or Jupyter
Notebook and run its cells in order.

The training notebook and Streamlit application must use the **same
model, tokenizer, and sequence-length configuration**. The
`lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` files should all
come from the same training run. Do not fit a new tokenizer for
inference.

The `qoute_dataset.csv` file is the dataset used for LSTM training. The
Streamlit application loads the saved model assets for prediction and
does not need to retrain the model every time it starts.

The GPT-2 notebook uses the pretrained model provided by Hugging Face.
It does not require or load the LSTM model assets.

## Saved LSTM Model Files

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

-   LSTM predictions depend on patterns learned from the quote dataset
    and may not always be grammatically or contextually appropriate.
-   The LSTM model can predict only words represented in its vocabulary.
-   The tokenizer and LSTM model must correspond to the same training
    run.
-   GPT-2's output is based on a pretrained general-purpose language
    model; it has not been fine-tuned for this project's quote dataset.
-   The GPT-2 notebook currently demonstrates one greedy next-token
    prediction rather than full text generation.
-   No accuracy score is claimed here. Add evaluation metrics only after
    measuring performance on held-out data.

## Future Improvements

-   Evaluate the LSTM model on a held-out test set.
-   Report Top-1 and Top-5 next-word prediction accuracy.
-   Experiment with sequence length, embedding dimensions, and LSTM
    units.
-   Compare the LSTM with simpler RNN and baseline language models.
-   Add temperature or Top-K sampling to make LSTM-generated text less
    deterministic.
-   Extend the GPT-2 notebook with an autoregressive generation loop and
    compare greedy decoding with sampling methods.
-   Compare the two approaches qualitatively using the same prompts,
    while clearly distinguishing the trained LSTM model from pretrained
    GPT-2.

## Author

**Parth Gupta**
