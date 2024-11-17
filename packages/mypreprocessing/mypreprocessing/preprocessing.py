from pandas import Series
import re
import nltk
import tensorflow as tf

def lower_case(series: Series):
    return series.str.lower()
    
def sanitize(series: Series):
    def sanitize_text(text: str):
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text) # special keywords
        text = re.sub(r'[^\w\s,]', '', text, flags=re.UNICODE) # emoticons
        text = re.sub(r'http\S+|www.\S+', '', text) # urls
        text = re.sub(r'\s+', ' ', text).strip() # multiple spaces
        return text

    return series.apply(sanitize_text)

def remove_stop_words(series: Series):
    stop_words = set(nltk.corpus.stopwords.words('english'))    
    def remove_from_single(text: str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    return series.apply(remove_from_single)

def tokenize_to_words(series: Series):        
    nltk.download('punkt')    
    return series.apply(nltk.tokenize.word_tokenize)

def stem_tokens(series):
    stemmer = nltk.stem.PorterStemmer()
    def stem_single(tokens):            
        return [stemmer.stem(token) for token in tokens]
    
    return series.apply(stem_single)

def count_unique_words(series: Series):
    all_tokens = [word for tokens in series for word in tokens]
    unique_words_count = len(set(all_tokens))
    return unique_words_count

def tokenize_to_numbers(series: Series, tokenizer):
    tokenizer.fit_on_texts(series)
    sequences = tokenizer.texts_to_sequences(series)
    return sequences

def save_tokenizer(tokenizer, filepath):
    with open(filepath, 'w') as f:
        f.write(tokenizer.to_json())

def load_tokenizer(filepath):
    with open(filepath) as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    
def trim_pad(series: Series, maxlen: int):
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
            series, 
            maxlen=maxlen, 
            padding='post', 
            truncating='post')
    
    return list(sequences)