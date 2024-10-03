import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import LatentDirichletAllocation
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import os
import numpy as np
from utils import *

# lematização e stopwords
nltk.download('stopwords')
nltk.download('wordnet')

class ReviewClassiffier:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    def load_data(self):
        """Carregar o dataset, selecionando colunas relevantes para o modelo"""
        print("-"*80)
        print(f"Section 1 - Loading dataset\n")
        print("Classifying reviews in the Amazon Products Reviews dataset\n")
        self.data = pd.read_csv(self.file_path, usecols=['primaryCategories', 'reviews.title', 'reviews.text',  'reviews.numHelpful', 'reviews.doRecommend', 'reviews.rating'])
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        print("-"*80)
    
    def clean_text(self, text):
        """Função para limpar e pré-processar o texto"""
        if pd.isna(text):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text) 
        text = text.lower()  
        text = text.split()  
        lemmatizer = WordNetLemmatizer()
        
        text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
        
        return ' '.join(text).strip() if text else ""

    def preprocess_data(self, output_path):
        """Aplicar pré-processamento em todo o dataset"""
        print(f"Section 2 - Starting preprocessing of data\n")
        # Remove duplicates e NaNs nas colunas 'reviews.text' e 'reviews.rating'
        self.data.drop_duplicates(subset=['reviews.text'], inplace=True)
        self.data.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)

        # Converter 'reviews.rating' para numérico, forçando erros a NaN e removendo-os
        self.data['reviews.rating'] = pd.to_numeric(self.data['reviews.rating'], errors='coerce')
        self.data.dropna(subset=['reviews.rating'], inplace=True)

        # Clean text and title
        print("Cleaning text and title...\n")
        self.data['cleaned_text'] = self.data['reviews.text'].apply(self.clean_text)
        self.data['cleaned_title'] = self.data['reviews.title'].apply(self.clean_text)

        # Substituir valores NaN em 'cleaned_text' e 'cleaned_title' por strings vazias
        self.data['cleaned_text'] = self.data['cleaned_text'].fillna('')
        self.data['cleaned_title'] = self.data['cleaned_title'].fillna('')

        # Combinar 'cleaned_title' e 'cleaned_text' para formar um único campo
        self.data['combined_text'] = self.data['cleaned_title'] + " " + self.data['cleaned_text']

        # Remover linhas com strings vazias ou NaN em 'combined_text'
        self.data['combined_text'] = self.data['combined_text'].fillna('').str.strip()
        self.data = self.data[self.data['combined_text'] != '']

        # Garantir que 'reviews.rating' é inteiro
        self.data['reviews.rating'] = self.data['reviews.rating'].astype(int)

        print(f"After preprocessing, shape of data: {self.data.shape}\n")
        
        # Column reviews.doRecommend is boolean, convert to binary
        self.data['reviews.doRecommend'] = self.data['reviews.doRecommend'].map({True: 1, False: 0})

        # Fill NaN values in reviews.numHelpful with 0
        self.data['reviews.numHelpful'] = self.data['reviews.numHelpful'].apply(lambda x: 0 if pd.isna(x) else x)

        # One-hot encode primaryCategories
        ohe = OneHotEncoder(sparse_output=False)
        categories_encoded = ohe.fit_transform(self.data[['primaryCategories']])
        self.data = pd.concat([self.data, pd.DataFrame(categories_encoded, columns=ohe.get_feature_names_out(['primaryCategories']))], axis=1)

        print("Data cleaned and features engineered successfully.\n")

        # Salvar o dataset pré-processado
        self.data.to_csv(output_path, index=False)

        print(f"Preprocessed data saved successfully at {os.path.abspath(output_path)}.")

        print("-"*80)

    def find_best_model(self, n_splits=5, n_repeats=10):
        """Pipeline para classificação de reviews usando diferentes modelos para encontrar o melhor"""
        print(f"Experimentation - Starting classification pipeline\n")

        texts = self.data['combined_text']
        ratings = self.data['reviews.rating']

        # Group ratings into sentiment categories
        y = ratings.replace({1: 'negative', 2: 'negative', 3: 'negative', 4: 'positive', 5: 'positive'}) # 1-3: negative, 4-5: positive
        y = y.map({'negative': 0, 'positive': 1})
        valid_indices = y.notnull() # Remove NaNs

        X = texts[valid_indices]
        y = y[valid_indices].astype(int)

        # Check class distribution
        print("Class distribution in y:")
        print(y.value_counts())

        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if n_splits > min_class_count:
            print(f"Ajustando n_splits de {n_splits} para {min_class_count} devido ao número mínimo de amostras em uma classe.")
            n_splits = min_class_count

        # Models
        # LogisticRegression - Média de balanced_accuracy_score was 0.87 after 10 repeats (Best model)
        model_lr = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        # BernoulliNB - Média de balanced_accuracy_score was 0.66 after 10 repeats
        model_nb = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', BernoulliNB())
        ])

        # RandomForest - Média de balanced_accuracy_score was 0.70 after 10 repeats
        model_rf = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', RandomForestClassifier(class_weight='balanced'))
        ])

        # # GradientBoosting - Média de balanced_accuracy_score was 0.69 after 10 repeats
        model_gb = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', GradientBoostingClassifier())
        ])

        # XGBoost - Média de balanced_accuracy_score was 0.87 after 10 repeats
        ratio = class_counts[0] / class_counts[1]
        model_xgb = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', XGBClassifier(scale_pos_weight=ratio))
        ])

        # LinearSVC - Média de balanced_accuracy_score was 0.83 after 10 repeats
        model_linearsvc = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', LinearSVC(class_weight='balanced', max_iter=10000))
        ])

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

        scores_nb = []
        scores_lr = []
        scores_rf = []
        scores_gb = []
        scores_xgb = []
        scores_linearsvc = []

        print(f"\nRunning classification pipeline with {n_splits}-fold cross-validation and {n_repeats} repeats...\n")

        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # LogisticRegression - Média de balanced_accuracy_score was 0.87 after 10 repeats (Best model)
            model_lr.fit(X_train, y_train)
            y_pred_lr = model_lr.predict(X_test)
            score_lr = balanced_accuracy_score(y_test, y_pred_lr)
            scores_lr.append(score_lr)

            # BernoulliNB - Média de balanced_accuracy_score was 0.66 after 10 repeats
            model_nb.fit(X_train, y_train)
            y_pred_nb = model_nb.predict(X_test)
            score_nb = balanced_accuracy_score(y_test, y_pred_nb)
            scores_nb.append(score_nb)

            # RandomForest - Média de balanced_accuracy_score was 0.70 after 10 repeats
            model_rf.fit(X_train, y_train)
            y_pred_rf = model_rf.predict(X_test)
            score_rf = balanced_accuracy_score(y_test, y_pred_rf)
            scores_rf.append(score_rf)

            # GradientBoosting - Média de balanced_accuracy_score was 0.69 after 10 repeats
            model_gb.fit(X_train, y_train)
            y_pred_gb = model_gb.predict(X_test)
            score_gb = balanced_accuracy_score(y_test, y_pred_gb)
            scores_gb.append(score_gb)

            # XGBoost - Média de balanced_accuracy_score was 0.87 after 10 repeats 
            model_xgb.fit(X_train, y_train)
            y_pred_xgb = model_xgb.predict(X_test)
            score_xgb = balanced_accuracy_score(y_test, y_pred_xgb)
            scores_xgb.append(score_xgb)

            # Linear SVC - Média de balanced_accuracy_score was 0.83 after 10 repeats
            model_linearsvc.fit(X_train, y_train)
            y_pred_linearsvc = model_linearsvc.predict(X_test)
            score_linearsvc = balanced_accuracy_score(y_test, y_pred_linearsvc)
            scores_linearsvc.append(score_linearsvc)

        # Output results
        print(f'\nMédia de balanced_accuracy_score para LogisticRegression após {n_repeats} repetições: {np.mean(scores_lr)}') # 0.87
        print(f'\nMédia de balanced_accuracy_score para BernoulliNB após {n_repeats} repetições: {np.mean(scores_nb)}')  # 0.66
        print(f'\nMédia de balanced_accuracy_score para RandomForest após {n_repeats} repetições: {np.mean(scores_rf)}')  # 0.70
        print(f'\nMédia de balanced_accuracy_score para GradientBoosting após {n_repeats} repetições: {np.mean(scores_gb)}')  # 0.69
        print(f'\nMédia de balanced_accuracy_score para XGBoost após {n_repeats} repetições: {np.mean(scores_xgb)}') # 0.87
        print(f'\nMédia de balanced_accuracy_score para LinearSVC após {n_repeats} repetições: {np.mean(scores_linearsvc)}') # 0.83
        print("-"*80)
    
    def classification_pipeline(self, n_splits=5, n_repeats=10):
        """Pipeline para classificação de reviews"""
        print(f"Section 3 - Starting classification pipeline\n")

        texts = self.data['combined_text']
        ratings = self.data['reviews.rating']

        # Group ratings into sentiment categories
        y = ratings.replace({1: 'negative', 2: 'negative', 3: 'negative', 4: 'positive', 5: 'positive'}) # 1-3: negative, 4-5: positive
        y = y.map({'negative': 0, 'positive': 1})
        valid_indices = y.notnull() # Remove NaNs

        X = texts[valid_indices]
        y = y[valid_indices].astype(int)

        # Check class distribution
        print("Class distribution in y:")
        print(y.value_counts())

        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if n_splits > min_class_count:
            print(f"Ajustando n_splits de {n_splits} para {min_class_count} devido ao número mínimo de amostras em uma classe.")
            n_splits = min_class_count

        # Models
        # LogisticRegression - Média de balanced_accuracy_score was 0.87 after 10 repeats (Best model)
        model_lr = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        # XGBoost - Média de balanced_accuracy_score was 0.87 after 10 repeats
        ratio = class_counts[0] / class_counts[1]
        model_xgb = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', XGBClassifier(scale_pos_weight=ratio))
        ])

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

        scores_lr = []
        scores_xgb = []

        # Listas para acumular y_test e y_pred de cada modelo
        all_y_test_lr = []
        all_y_pred_lr = []

        all_y_test_xgb = []
        all_y_pred_xgb = []

        # Curvas de aprendizado
        print("\nCurvas de Aprendizado:")
        cv_learning_curve = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=42)

        train_sizes, train_scores_mean, test_scores_mean = plot_learning_curve(
            model=model_lr,
            X=X,
            y=y,
            cv=cv_learning_curve,
            scoring='balanced_accuracy',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        # Análise das curvas
        print("\nAnálise das Curvas de Aprendizado:")
        print("Scores de Treinamento Médios:", train_scores_mean)
        print("Scores de Validação Médios:", test_scores_mean)

        print(f"\nRunning classification pipeline with {n_splits}-fold cross-validation and {n_repeats} repeats...\n")

        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # LogisticRegression - Média de balanced_accuracy_score was 0.87 after 10 repeats (Best model)
            model_lr.fit(X_train, y_train)
            y_pred_lr = model_lr.predict(X_test)
            score_lr = balanced_accuracy_score(y_test, y_pred_lr)
            scores_lr.append(score_lr)

            all_y_test_lr.extend(y_test.tolist())
            all_y_pred_lr.extend(y_pred_lr.tolist())

            # XGBoost - Média de balanced_accuracy_score was 0.87 after 10 repeats 
            model_xgb.fit(X_train, y_train)
            y_pred_xgb = model_xgb.predict(X_test)
            score_xgb = balanced_accuracy_score(y_test, y_pred_xgb)
            scores_xgb.append(score_xgb)

            all_y_test_xgb.extend(y_test.tolist())
            all_y_pred_xgb.extend(y_pred_xgb.tolist())

        # Output results
        print(f'\nMédia de balanced_accuracy_score para LogisticRegression após {n_repeats} repetições: {np.mean(scores_lr)}') # 0.87
        print(f'\nMédia de balanced_accuracy_score para XGBoost após {n_repeats} repetições: {np.mean(scores_xgb)}') # 0.87

        vectorizer = model_lr.named_steps['vectorizer']
        classifier = model_lr.named_steps['classifier']
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]

        # Create a DataFrame with words and their coefficients
        coef_df = pd.DataFrame({'word': feature_names, 'coefficient': coefficients})

        # Top words contributing to the positive class
        top_positive = coef_df.sort_values(by='coefficient', ascending=False).head(20)

        # Top words contributing to the negative class
        top_negative = coef_df.sort_values(by='coefficient').head(20)

        print("\nTop words contributing to positive reviews:")
        print(top_positive.to_string(index=False))

        print("\nTop words contributing to negative reviews:")
        print(top_negative.to_string(index=False))

        # Matriz de confusão para LogisticRegression
        cm_lr = confusion_matrix(all_y_test_lr, all_y_pred_lr, labels=[0, 1])
        cm_df_lr = pd.DataFrame(cm_lr, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
        print("\nConfusion Matrix for LogisticRegression:")
        print(cm_df_lr)

        # Matriz de confusão para XGBoost
        cm_xgb = confusion_matrix(all_y_test_xgb, all_y_pred_xgb, labels=[0, 1])
        cm_df_xgb = pd.DataFrame(cm_xgb, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
        print("\nConfusion Matrix for XGBoost:")
        print(cm_df_xgb)

        print("-"*80)

    def topic_analysis(self, n_topics=5):
        """Performs topic analysis and evaluates classification performance by topic."""
        print(f"Section 5 - Topic Analysis\n")

        texts = self.data['combined_text']
        ratings = self.data['reviews.rating']

        # Map ratings directly to 0 (negative) and 1 (positive)
        y = ratings.replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        valid_indices = y.notnull()  # Remove NaNs

        X = texts[valid_indices]
        y = y[valid_indices].astype(int)

        print("Performing Topic Modeling...")
        vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        X_vectors = vectorizer.fit_transform(X)

        # Fit the LDA model
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_model.fit(X_vectors)

        # Display the topics with their top words
        feature_names = vectorizer.get_feature_names_out()
        def display_topics(model, feature_names, no_top_words):
            for topic_idx, topic in enumerate(model.components_):
                print(f"\nTopic {topic_idx+1}:")
                print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

        display_topics(lda_model, feature_names, no_top_words=10)

        # Assign topics to documents
        topic_distribution = lda_model.transform(X_vectors)
        document_topics = np.argmax(topic_distribution, axis=1)
        self.data = self.data.loc[valid_indices]
        self.data['topic'] = document_topics

        print("\nAnalyzing Classification Performance by Topic...")
        topic_performance = pd.DataFrame(columns=['Topic', 'Balanced Accuracy'])

        for topic_num in range(n_topics):
            topic_indices = self.data[self.data['topic'] == topic_num].index
            X_topic = self.data.loc[topic_indices, 'combined_text']
            y_topic = self.data.loc[topic_indices, 'reviews.rating'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})

            if len(y_topic.unique()) < 2 or len(y_topic) < 100:
                print(f"Skipping Topic {topic_num+1} due to insufficient data.")
                continue

            # Define the classifier
            model = Pipeline([
                ('vectorizer', CountVectorizer(binary=True)),
                ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
            ])

            # Cross-validation
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            scores = cross_val_score(model, X_topic, y_topic, cv=cv, scoring='balanced_accuracy', n_jobs=-1)

            # Store the results
            new_row = pd.DataFrame({
                'Topic': [topic_num+1],
                'Balanced Accuracy': [np.mean(scores)]
            })
            topic_performance = pd.concat([topic_performance, new_row], ignore_index=True)


        print("\nClassification Performance by Topic:")
        print(topic_performance)

        print("\nImplementing Two-Layer Classifier...")
        topic_classifiers = {}

        for topic_num in range(n_topics):
            topic_indices = self.data[self.data['topic'] == topic_num].index
            X_topic = self.data.loc[topic_indices, 'combined_text']
            y_topic = self.data.loc[topic_indices, 'reviews.rating'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})

            # Ensure there are enough samples
            if len(y_topic.unique()) < 2 or len(y_topic) < 100:
                print(f"Skipping Topic {topic_num+1} due to insufficient data.")
                continue

            # Define the classifier
            model = Pipeline([
                ('vectorizer', CountVectorizer(binary=True)),
                ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
            ])

            # Train the classifier
            model.fit(X_topic, y_topic)

            # Store the classifier
            topic_classifiers[topic_num] = model

        # Evaluate the Two-Layer Classifier
        all_y_true = []
        all_y_pred = []

        for idx, row in self.data.iterrows():
            text = row['combined_text']
            true_label = 0 if row['reviews.rating'] <= 3 else 1

            # First Layer: Topic Assignment
            topic = row['topic']

            if topic in topic_classifiers:
                # Second Layer: Sentiment Prediction
                classifier = topic_classifiers[topic]
                pred_label = classifier.predict([text])[0]
            else:
                # Default Prediction
                pred_label = 1  # Assuming positive is the majority

            all_y_true.append(true_label)
            all_y_pred.append(pred_label)

        overall_balanced_accuracy = balanced_accuracy_score(all_y_true, all_y_pred)
        print(f"\nOverall Balanced Accuracy for Two-Layer Classifier: {overall_balanced_accuracy}")

        print("\nComparing with Single-Layer Classifier...")
        # Single-layer classifier (using the entire dataset)
        model_lr = Pipeline([
            ('vectorizer', CountVectorizer(binary=True)),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        scores_lr = cross_val_score(model_lr, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        single_layer_accuracy = np.mean(scores_lr)
        print(f"Balanced Accuracy for Single-Layer Classifier: {single_layer_accuracy}")

        print("-" * 80)

        print(f"The overall balanced accuracy of the two-layer classifier is {overall_balanced_accuracy:.4f},")
        print(f"compared to {single_layer_accuracy:.4f} for the single-layer classifier.")

        if overall_balanced_accuracy > single_layer_accuracy:
            print("The two-layer classifier outperforms the single-layer classifier.")
        else:
            print("The two-layer classifier does not outperform the single-layer classifier.")



if __name__ == "__main__":
    file_path = 'data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'  # Caminho para o dataset input
    output_prepro = 'data/prepro_dataset.csv'  # Caminho para salvar o dataset pré-processado
    classifier = ReviewClassiffier(file_path)
    classifier.load_data() # Section 1 - Loading dataset
    classifier.preprocess_data(output_prepro) # Section 2 - Preprocessing data
    experimentation = False
    if experimentation:
        classifier.find_best_model(n_splits=5, n_repeats=10) # Experimentation - Classification pipeline to find the best model
    else:
        classifier.classification_pipeline(n_splits=5, n_repeats=10) # Section 3 & 4 - Classification pipeline to classify reviews & Dataset analysis with learning curves
        classifier.topic_analysis(n_topics=5)
    



# Confusion Matrix for LogisticRegression:
#                  Predicted Negative  Predicted Positive
# Actual Negative               15950                4070
# Actual Positive               10267              151373

# Confusion Matrix for XGBoost:
#                  Predicted Negative  Predicted Positive
# Actual Negative               16701                3319
# Actual Positive               15771              145869


