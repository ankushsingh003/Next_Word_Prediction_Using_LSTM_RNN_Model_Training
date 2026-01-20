# Next Word Prediction using LSTM

This project implements a next word prediction model using Long Short-Term Memory (LSTM) networks trained on Shakespeare's Hamlet text. The model predicts the next word in a sequence based on the input text. It includes a Jupyter notebook for training and a Streamlit web app for interactive predictions.

## Features

- Pre-trained LSTM model for next word prediction
- Web-based interface using Streamlit for easy prediction
- Jupyter notebook for training the model from scratch
- Uses Shakespeare's Hamlet as training data

## Files

- `Experiments.ipynb`: Jupyter notebook for data preparation, model training, and testing predictions
- `app.py`: Streamlit application for interactive next word prediction
- `hamlet.txt`: The training text data (Shakespeare's Hamlet)
- `next_word_lstm_model.h5`: Pre-trained LSTM model file
- `tokenizer_hamlet.pickle`: Pickled tokenizer for text preprocessing
- `requirements.txt`: Python dependencies

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone or download this repository to your local machine.

2. Navigate to the project directory:
   ```
   cd path/to/LSTM_GRU
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   This will install:
   - tensorflow
   - pandas
   - numpy
   - scikit-learn
   - nltk
   - tensorboard
   - matplotlib
   - seaborn
   - streamlit
   - scikeras

4. Download NLTK data (if not already done):
   ```python
   import nltk
   nltk.download('gutenberg')
   ```

## Usage

### Training the Model (Optional)

If you want to retrain the model:

1. Open `Experiments.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells in the notebook. This will:
   - Download and preprocess the Hamlet text
   - Tokenize the text and create input sequences
   - Build and train the LSTM model
   - Save the trained model and tokenizer

### Running the Prediction App

1. Ensure the model and tokenizer files are in the same directory as `app.py`.

2. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to the URL displayed (usually `http://localhost:8501`).

4. Enter some text in the text area and click "Predict Next Word" to see the model's prediction.

### Testing Predictions (Alternative)

1. Open `Experiments.ipynb` in Jupyter Notebook.
2. Run the cells to load the model and test the prediction function.
3. Modify the `input_text` variable in the test cells to try different inputs.

## Model Details

- **Architecture**: Embedding layer -> LSTM (150 units, return sequences) -> Dropout (0.2) -> LSTM (100 units) -> Dense output layer
- **Input**: Text sequences (padded/truncated to 50 words)
- **Output**: Predicted next word from the vocabulary
- **Dataset**: Shakespeare's Hamlet text from NLTK Gutenberg corpus
- **Vocabulary size**: ~4,800 unique words
- **Training**: 10 epochs with Adam optimizer and sparse categorical crossentropy loss

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Model loading issues**: Make sure `next_word_lstm_model.h5` and `tokenizer_hamlet.pickle` are in the same directory as the scripts
- **Streamlit not starting**: Check that port 8501 is available or specify a different port with `streamlit run app.py --server.port 8502`
- **Memory issues during training**: If training fails due to memory, reduce batch size (default is 32) or sequence length
- **NLTK download issues**: Run `python -c "import nltk; nltk.download('gutenberg')"` to download the corpus manually

## License

This project is for educational purposes. The Hamlet text is from the NLTK Gutenberg corpus.