import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load datasets
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['Index', 'Category', 'Sentiment', 'Text']
    return df.drop(columns=['Index', 'Category'])  # Drop unnecessary columns

twitter_train = load_data('./twitter_training.csv')
twitter_validation = load_data('./twitter_validation.csv')

# Combine datasets
data = pd.concat([twitter_train, twitter_validation], ignore_index=True)

# Drop missing values
data.dropna(subset=['Text', 'Sentiment'], inplace=True)

# Standardize sentiment labels
data['Sentiment'] = data['Sentiment'].str.lower().str.strip()

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    words = word_tokenize(str(text).lower())
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

data['Clean_Text'] = data['Text'].apply(preprocess_text)

# WordCloud visualization
sentiments = data['Sentiment'].unique()
fig, axes = plt.subplots(len(sentiments), 1, figsize=(10, 6 * len(sentiments)))

for i, sentiment in enumerate(sentiments):
    text_data = ' '.join(data[data['Sentiment'] == sentiment]['Clean_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(f'WordCloud for Sentiment: {sentiment}')
    axes[i].axis('off')

plt.show()

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
X = vectorizer.fit_transform(data['Clean_Text'])
y = data['Sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier with hyperparameter tuning
param_grid = {'alpha': [0.1, 0.5, 1.0]}
nb_model = GridSearchCV(MultinomialNB(), param_grid, cv=5)
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate model
print("Best Parameters:", nb_model.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the trained model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(nb_model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
