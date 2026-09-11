from nltk.stem import PorterStemmer
import string

def tokenize_term(term: str) -> str:
    tokens = preprocess(term)
    if len(tokens) != 1:
        raise ValueError(f"Expected exactly one token, got {len(tokens)}: {tokens}")
    return tokens[0]

def remove_punctuation(text:str) -> str:
    # string.punctuation is just a string containing all punctuation characters: !"#$%&'()*+,-./:;<=>?@[\]^_{|}~
    # str.maketrans("", "", string.punctuation) builds a translation table that maps every punctuation character to None (i.e. delete it)
    # .translate(table) applies that table to the string
    return text.translate(str.maketrans("", "", string.punctuation))


def load_stop_words(file_path: str) -> set:
    with open(file_path) as f:
        stop_words = f.read().splitlines()
    # preprocess each stop word the same way you preprocess query/title tokens
    return set(remove_punctuation(word.lower()) for word in stop_words)

# load once at module level so you don't re-read the file every call
STOP_WORDS = load_stop_words("data/rag_visual/stop_words.txt")


def preprocess(query: str) -> str:
    # Remove Case Sensitivity
    lower_ = query.lower()
    # Remove Punctuation
    punc_ = remove_punctuation(lower_)
    # Tokenization
    token_ = punc_.split()
    # Stop Words
    token_ = [word for word in token_ if word not in STOP_WORDS]
    # stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in token_]
    return stemmed