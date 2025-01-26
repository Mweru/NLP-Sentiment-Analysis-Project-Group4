import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.decomposition import PCA
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DataUnderstanding:
    def __init__(self, df):
        self.df = df
        
    def first_five_rows(self):
        print("\nFirst five rows:")
        print(self.df.head())

    def last_five_rows(self):
        print("\nLast five rows:")
        print(self.df.tail())

    def basic_info(self):
        print("\nBasic Information:")
        print(self.df.info())

    def basic_shape(self):
        print("\nBasic Shape:")
        print(self.df.shape)
        
    def statistical_summary(self):
        print("\nStatistical Summary:")
        print(self.df.describe())

class DataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe

    def shorten_column_names(self):
        self.df.columns = ['tweet', 'brand_product', 'emotion']

    def check_missing_values(self):
        return self.df.isnull().sum()

    def check_value_counts(self, column_name):
        return np.array(self.df[column_name].value_counts().index)

    def handle_missing_brand_product(self, categories):
        
        for i, row in self.df.iterrows():
            if pd.isnull(row['brand_product']):
      
                tweet = str(row['tweet']) if pd.notnull(row['tweet']) else ""
                
                for category in np.concatenate((categories, np.char.lower(categories))):
                    if category in tweet.lower():
                        self.df.loc[i, 'brand_product'] = category
                        break
            
    def drop_missing_values(self):
        self.df = self.df.dropna()

    def check_duplicates(self):
        return self.df.duplicated().sum()

    def drop_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self

    
class TextPreprocessor:
    def __init__(self, df, stop_words_language='english'):
        self.df = df
        self.stop_words = set(stopwords.words(stop_words_language))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def clean_text(self, text):
        """
        Clean the text by removing URLs, mentions, hashtags, non-alphanumeric characters, and extra spaces.
        """
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^a-zA-Z0-9\s]', '', text)  # Remove URLs, mentions, and hashtags
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    def tokenize(self, text):
        """
        Tokenize the text into words.
        """
        return word_tokenize(text)

    def to_lowercase(self, tokens):
        """
        Convert tokens to lowercase.
        """
        return [token.lower() for token in tokens]

    def remove_stopwords_and_short_words(self, tokens):
        """
        Remove stopwords and short words (less than 3 characters) from the token list.
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 2 and not token.isdigit()]

    def get_wordnet_pos(self, treebank_tag):
        """
        Convert the treebank POS tag to a WordNet POS tag.
        """
        if treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN  # Default to NOUN if no other match

    def lemmatize_with_pos(self, tokens):
        """
        Lemmatize tokens using POS tagging to improve lemmatization accuracy.
        """
        tagged_tokens = pos_tag(tokens)
        return [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in tagged_tokens]

    def preprocess(self):
        """
        Apply all preprocessing steps: cleaning, tokenization, removing stopwords, and lemmatization.
        """
        self.df['cleaned_tweet_text'] = self.df['tweet'].apply(self.clean_text)
        self.df['tokens'] = self.df['cleaned_tweet_text'].apply(self.tokenize)
        self.df['tokens_lowercased'] = self.df['tokens'].apply(self.to_lowercase)
        self.df['tokens_stopwords_removed'] = self.df['tokens_lowercased'].apply(self.remove_stopwords_and_short_words)
        self.df['lemmatized_text'] = self.df['tokens_stopwords_removed'].apply(self.lemmatize_with_pos)

    def vectorize(self, texts):
        texts_joined = [' '.join(tokens) for tokens in texts]
        X = self.vectorizer.fit_transform(texts_joined)
        return pd.DataFrame(X.toarray(), columns=self.vectorizer.get_feature_names_out())

    def get_processed_data(self):
        """
        Return the DataFrame with all processed columns.
        """
        return self.df

class Encoder:
    def __init__(self, dataframe):
        self.df = dataframe
        
    def encode(self, column, column_map):
        self.df[column] = self.df[column].map(column_map)
        return self.df
        

class EDA:
    def __init__(self, df):
        self.df = df

    def plot_emotion_distribution(self):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        binary = self.df['emotion'].value_counts().nsmallest(2).index
        sns.countplot(data=self.df, x='emotion', order=binary, palette='Set2')
        plt.title('Distribution of Emotions (Binary)')
        plt.xlabel('Emotion Binary')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        sns.countplot(data=self.df, x='emotion', palette='Set2')
        plt.title('Distribution of Emotions (Multi-Class)')
        plt.xlabel('Emotion Multi-Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_brand_distribution(self):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 2)
        sns.countplot(data=self.df, x='brand_product', palette='Set3', 
                      order=self.df['brand_product'].value_counts().index)
        plt.title('Brand/Product Mentions')
        plt.xlabel('Brand/Product')
        plt.ylabel('Count')
        plt.xticks(rotation=90)

        plt.subplot(1, 2, 1)
        sns.countplot(data=self.df, x='brand_category', palette='Set3', 
                      order=self.df['brand_category'].value_counts().index)
        plt.title('Brand Category Mentions')
        plt.xlabel('Brand Category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_tweet_length_distribution(self):
        self.df['tweet_length'] = self.df['tweet'].apply(lambda x: len(str(x)))  # Convert x to string first
        plt.figure(figsize=(8, 6))
        sns.histplot(self.df['tweet_length'], kde=True, color='skyblue')
        plt.title('Distribution of Tweet Lengths')
        plt.xlabel('Tweet Length (Words)')
        plt.ylabel('Count')
        plt.show()

    def plot_top_words(self):
        all_text = ' '.join(self.df['lemmatized_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))
    
        # Split the text into words and count them
        words = all_text.split()
        word_counts = Counter(words)
        most_common_words = word_counts.most_common(10)
        words, counts = zip(*most_common_words)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words), palette='Set3')
        plt.title('Top 10 Most Frequent Words in Tweets')
        plt.xlabel('Count')
        plt.ylabel('Words')
        plt.show()

    def plot_sentiment_vs_brand(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='brand_category', hue='emotion', palette='Set2',
                      order=self.df['brand_category'].value_counts().index)
        plt.title('Sentiment vs Brand Category Mention')
        plt.xlabel('Brand Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Emotion')
        plt.show()

    def plot_tweet_length_by_sentiment(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.df, x='emotion', y='tweet_length', palette='Set2')
        plt.title('Sentiment vs Tweet Length')
        plt.xlabel('Emotion')
        plt.ylabel('Tweet Length (Words)')
        plt.show()

    def plot_wordclouds_by_emotion(self):
        plt.figure(figsize=(14, 6))
        unique_emotions = self.df['emotion'].unique()

        for i, emotion in enumerate(unique_emotions):
            # Combine the lemmatized_text for all rows of the given emotion into a single string
            emotion_text = ' '.join(self.df[self.df['emotion'] == emotion]['lemmatized_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))                                                                                         
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(emotion_text)
            plt.subplot(1, len(unique_emotions), i + 1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Most Frequent Words: {emotion}')

        plt.tight_layout()
        plt.show()

    def plot_wordclouds_by_emotion_and_brand(self):
        plt.figure(figsize=(14, 8))
        unique_emotions = self.df['emotion'].unique()
        for i, emotion in enumerate(unique_emotions):
            for j, brand in enumerate(['apple', 'google']):
                # Ensure lemmatized_text is a string even if it's a list
                emotion_and_brand_text = ' '.join(
                    self.df[(self.df['emotion'] == emotion) & 
                            (self.df['brand_category'].str.lower() == brand)]['lemmatized_text'].apply(
                                lambda x: ' '.join(x) if isinstance(x, list) else str(x))
                )
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(emotion_and_brand_text)
                
                # Plot the word cloud for each emotion and brand
                plt.subplot(len(unique_emotions), 2, i * 2 + j + 1)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Emotion: {emotion}, Brand: {brand.capitalize()}')

        plt.tight_layout()
        plt.show()

    def plot_pairplot_sentiment_analysis(self):
        pairplot = sns.pairplot(self.df, vars=['tweet_length'], hue='emotion', 
                                palette='Set2', kind='scatter', plot_kws={'alpha': 0.7})
        plt.suptitle('Sentiment, Brand/Product and Tweet Length Analysis', y=1.02)
        pairplot.fig.legend(title='Emotion', loc='upper right', bbox_to_anchor=(1, 1))
        plt.show()

    def plot_pca_analysis(self, X_df):
        binary = self.df['emotion'].value_counts().nsmallest(2).index
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_df)
        pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue=self.df['emotion'], 
                        palette='Set2', style=self.df['emotion'], hue_order = binary, s=100, ax=axes[0])
        axes[0].set_title('PCA of TF-IDF Features - Emotion Binary')
        axes[0].set_xlabel('PCA1')
        axes[0].set_ylabel('PCA2')
        axes[0].legend(title='Emotion Binary')
         
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue=self.df['emotion'], 
                        palette='Set2', style=self.df['emotion'], s=100, ax=axes[1])
        axes[1].set_title('PCA of TF-IDF Features - Emotion')
        axes[1].set_xlabel('PCA1')
        axes[1].set_ylabel('PCA2')
        axes[1].legend(title='Emotion')

        plt.tight_layout()
        plt.show()
class SentimentModel:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(),
            'svm': SVC(),
            'random_forest': RandomForestClassifier(),
            'xgboost': xgb.XGBClassifier()
        }

    def preprocess_data(self, X_train, X_test, apply_scaling=True, apply_pca=True):
        # Apply scaling
        if apply_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Apply PCA for dimensionality reduction
        if apply_pca:
            pca = PCA(n_components=0.95)  # Preserve 95% variance
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        return X_train, X_test

    def apply_smote(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled

    def train(self, X_train, X_test, y_train, y_test, task_type='binary', apply_scaling=True, apply_pca=True):
        # Preprocess data (scaling and PCA)
        X_train, X_test = self.preprocess_data(X_train, X_test, apply_scaling, apply_pca)

        # Apply SMOTE for class imbalance (for both binary and multi-class tasks)
        X_train, y_train = self.apply_smote(X_train, y_train)

        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"Training model: {model_name}")

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate and display results
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{model_name.capitalize()} - Accuracy: {accuracy:.4f}")
            print(f"{model_name.capitalize()} - Classification Report:")
            print(classification_report(y_test, y_pred))
            print(f"{model_name.capitalize()} - Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("-" * 50)

class BertModel:
    def __init__(self, data_path, model_type="distilbert-base-uncased", max_samples=3000, mode="binary"):
        """
        Initialize the BertModel class.
        Args:
            data_path (str): Path to the dataset.
            model_type (str): Hugging Face model type. Default is 'distilbert-base-uncased'.
            max_samples (int): Maximum number of samples to use for faster training. Default is 3000.
            mode (str): "binary" for binary classification or "multi" for multi-class classification.
        """
        self.data_path = data_path
        self.model_type = model_type
        self.max_samples = max_samples
        self.mode = mode

        # Load tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_type)

    def load_and_process_data(self):
        """
        Load and process data by balancing classes and limiting sample size.
        """
        # Load dataset
        df = pd.read_csv(self.data_path)
        df = df.sample(self.max_samples, random_state=42)  # Limit dataset size

        if self.mode == "binary":
            # Binary classification: Balance Positive and Negative classes
            df = df[df['emotion'].isin(['Positive', 'Negative'])]
            df['label'] = df['emotion'].map({'Negative': 0, 'Positive': 1})

            negative_df = df[df['label'] == 0]
            positive_df = df[df['label'] == 1]
            positive_upsampled = resample(positive_df, replace=True, n_samples=len(negative_df), random_state=42)
            df_balanced = pd.concat([negative_df, positive_upsampled])

        elif self.mode == "multi":
            # Multi-class classification: Balance Negative, Neutral, and Positive classes
            df['label'] = df['emotion'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})
            class_0 = df[df['label'] == 0]
            class_1 = df[df['label'] == 1]
            class_2 = df[df['label'] == 2]

            max_class_size = max(len(class_0), len(class_1), len(class_2))
            class_0_upsampled = resample(class_0, replace=True, n_samples=max_class_size, random_state=42)
            class_1_upsampled = resample(class_1, replace=True, n_samples=max_class_size, random_state=42)
            class_2_upsampled = resample(class_2, replace=True, n_samples=max_class_size, random_state=42)
            df_balanced = pd.concat([class_0_upsampled, class_1_upsampled, class_2_upsampled])

        else:
            raise ValueError("Invalid mode. Use 'binary' or 'multi'.")

        # Train-test split
        train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
        self.train_texts = list(train_df['tweet'])
        self.test_texts = list(test_df['tweet'])
        self.train_labels = list(train_df['label'])
        self.test_labels = list(test_df['label'])

    def tokenize_data(self):
        """
        Tokenize the text data for the model.
        """
        self.train_encodings = self.tokenizer(self.train_texts, truncation=True, padding=True, max_length=128)
        self.test_encodings = self.tokenizer(self.test_texts, truncation=True, padding=True, max_length=128)

    def build_model(self):
        """
        Build the DistilBERT model for classification.
        """
        num_labels = 2 if self.mode == "binary" else 3
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_type, num_labels=num_labels)

    def train_model(self, epochs=3, batch_size=8):
        """
        Train the model.
        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        class EmotionDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        train_dataset = EmotionDataset(self.train_encodings, self.train_labels)
        test_dataset = EmotionDataset(self.test_encodings, self.test_labels)

        # Optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        num_train_steps = len(train_dataset) * epochs
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_train_steps
        )

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="tensorboard"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            optimizers=(optimizer, scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping
        )
        self.trainer.train()

    def evaluate_model(self):
        """
        Evaluate the model and display metrics.
        """
        preds = self.trainer.predict(self.trainer.eval_dataset)
        preds_labels = torch.argmax(torch.tensor(preds.predictions), axis=1).numpy()

        print(f"Classification Report ({self.mode} classification):")
        print(classification_report(self.test_labels, preds_labels))

        print(f"Confusion Matrix ({self.mode} classification):")
        print(confusion_matrix(self.test_labels, preds_labels))
