import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load datasets
twitter_train = pd.read_csv('./twitter_training.csv', header=None)
twitter_validation = pd.read_csv('./twitter_validation.csv', header=None)

# Assign column names based on observed structure
twitter_train.columns = ['Index', 'Category', 'Sentiment', 'Text']
twitter_validation.columns = ['Index', 'Category', 'Sentiment', 'Text']

# Combine datasets
data = pd.concat([twitter_train, twitter_validation], ignore_index=True)

# Check data structure
print(data.head())

# Sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='Sentiment', palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Text preprocessing
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(str(text).lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

data['Clean_Text'] = data['Text'].apply(preprocess_text)

# WordCloud for each sentiment
sentiments = data['Sentiment'].unique()
fig, axes = plt.subplots(len(sentiments), 1, figsize=(10, 6 * len(sentiments)))

for i, sentiment in enumerate(sentiments):
    text_data = ' '.join(data[data['Sentiment'] == sentiment]['Clean_Text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(f'WordCloud for Sentiment: {sentiment}')
    axes[i].axis('off')

plt.show()

# Convert text data to numerical features using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2))
X = tfidf_vectorizer.fit_transform(data['Clean_Text'])
y = data['Sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save processed dataset
data.to_csv('./processed_twitter_data.csv', index=False)
