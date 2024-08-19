import pandas as pd
import numpy as np
import math
import random 
import os 
import sys 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

SENTIMENTS = ["Positive", "Neutral", "Negative"]
directory_path = "Output"
quality = 130 

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory
    os.makedirs(directory_path)

def get_vocabulary(df, apply_lemmatizing):
    vocabulary = set()
    vocabulary_key = {}

    for tweet in df['CoronaTweet']:

        for word in tweet.split():
            if apply_lemmatizing:
                lemmatized_word = part_d_perform_lemmatizing(word)
                if word.lower() not in stop_words and lemmatized_word not in vocabulary:
                    vocabulary.add(lemmatized_word)
                    vocabulary_key[lemmatized_word] = len(vocabulary) - 1
            else:
                if word not in vocabulary:
                    vocabulary.add(word)
                    vocabulary_key[word] = len(vocabulary) - 1

    sorted_vocabulary = sorted(vocabulary)
    return np.array(sorted_vocabulary), vocabulary_key


def training(file_path, apply_lemmatizing, file_path2):
    df = pd.read_csv(file_path)
    if 'CoronaTweet' not in df.columns:
        df.rename(columns={'Tweet': 'CoronaTweet'}, inplace=True)

    if file_path2 != None:
        df2 = pd.read_csv(file_path2)
        df2.rename(columns={'Tweet': 'CoronaTweet'}, inplace=True)
        df = pd.concat([df, df2])

    y = []
    x = []
    vocabulary, vocabulary_key = get_vocabulary(df, apply_lemmatizing)
    sentiment_word_occur = {sentiment: {} for sentiment in SENTIMENTS}
    word_cloud_sentiment_map = {sentiment: {} for sentiment in SENTIMENTS}

    for index, row in df.iterrows():
        sentiment = row['Sentiment']
        tweet = row['CoronaTweet']

        tweet_x = [vocabulary_key[word] for word in tweet.split() if word in vocabulary_key]
        x.append(tweet_x)
        y.append(sentiment)

        for word in tweet.split():
            if apply_lemmatizing:
                lemmatized_word = part_d_perform_lemmatizing(word)
                if lemmatized_word in vocabulary_key and word not in stop_words and lemmatized_word not in stop_words:
                    position = vocabulary_key[lemmatized_word]
                    sentiment_word_occur[sentiment][position] = sentiment_word_occur[sentiment].get(position, 0) + 1
                    word_cloud_sentiment_map[sentiment][lemmatized_word] = word_cloud_sentiment_map[sentiment].get(lemmatized_word, 0) + 1

            else:
                if word in vocabulary_key:
                    position = vocabulary_key[word]
                    sentiment_word_occur[sentiment][position] = sentiment_word_occur[sentiment].get(position, 0) + 1
                    word_cloud_sentiment_map[sentiment][word] = word_cloud_sentiment_map[sentiment].get(word, 0) + 1

    sentiment_total = {sentiment: (np.count_nonzero(np.array(y) == sentiment)) for sentiment in SENTIMENTS}

    return x, np.array(y), vocabulary, vocabulary_key, sentiment_word_occur, sentiment_total, word_cloud_sentiment_map

def get_word_probability_by_position(sentiment_word_occur, position, sentiment, sentiment_total):
    return (sentiment_word_occur[sentiment].get(position, 0) + 1) / (sentiment_total[sentiment] + 3)

def calculate_sentiment_probabilities(x, y, vocabulary, vocabulary_key, sentiment_word_occur, sentiment_total, tweet, apply_lemmatizing):

    sentiment_probability = {}

    for sentiment in SENTIMENTS:
        sentiment_probability[sentiment] = math.log(sentiment_total[sentiment] / np.size(y))

        for word in tweet.split():
            if apply_lemmatizing:
                if word.lower() in stop_words: 
                    continue
                lemmatized_word = part_d_perform_lemmatizing(word)
                try:
                    sentiment_probability[sentiment] += math.log(get_word_probability_by_position(sentiment_word_occur, vocabulary_key[lemmatized_word], sentiment, sentiment_total))
                except:
                    sentiment_probability[sentiment] += math.log(1 / (sentiment_total[sentiment] + 3))
                    
            else:
                try:
                    sentiment_probability[sentiment] += math.log(get_word_probability_by_position(sentiment_word_occur, vocabulary_key[word], sentiment, sentiment_total))
                except:
                    sentiment_probability[sentiment] += math.log(1 / (sentiment_total[sentiment] + 3))

    log_total_probability = max(sentiment_probability.values())
    for sentiment in SENTIMENTS:
        sentiment_probability[sentiment] -= log_total_probability

    total_probability = sum(math.exp(sentiment_probability[sentiment]) for sentiment in SENTIMENTS)
    for sentiment in SENTIMENTS:
        sentiment_probability[sentiment] = math.exp(sentiment_probability[sentiment]) / total_probability

    max_sentiment = max(sentiment_probability, key=sentiment_probability.get)

    return sentiment_probability, max_sentiment

def create_word_cloud(word_cloud_sentiment_map, set_type, png_file_path):
    for sentiment in SENTIMENTS:
        word_cloud_map = word_cloud_sentiment_map[sentiment]

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_cloud_map)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        if not os.path.exists(f"{directory_path}/{png_file_path}/"):
            os.makedirs(f"{directory_path}/{png_file_path}/")

        plt.savefig(f"{directory_path}/{png_file_path}/wordcloud_{set_type}_{sentiment}.png", dpi = quality)
        plt.close()

def part_a(tweets_training_file_path, tweets_validation_file_path, set_type, txt_file_path, word_cloud, apply_lemmatizing = False, tweets_training_file_path2 = None):

    x, y, vocabulary, vocabulary_key, sentiment_word_occur, sentiment_total, word_cloud_sentiment_map = training(tweets_training_file_path, apply_lemmatizing, tweets_training_file_path2)

    df_validation = pd.read_csv(tweets_validation_file_path)
    if 'CoronaTweet' not in df_validation.columns:
        df_validation.rename(columns={'Tweet': 'CoronaTweet'}, inplace=True)

    accuracy = 0
    validation_total = len(df_validation)
    confusion_matrix = {sentiment: {sentiment: 0 for sentiment in SENTIMENTS} for sentiment in SENTIMENTS}

    for index, row in df_validation.iterrows():
        sentiment = row['Sentiment']
        sentiment_probability, max_sentiment = calculate_sentiment_probabilities(x, y, vocabulary, vocabulary_key, sentiment_word_occur, sentiment_total, row['CoronaTweet'], apply_lemmatizing)
        if sentiment == max_sentiment:
            accuracy += 1
        confusion_matrix[max_sentiment][sentiment] += 1

    print(f"{set_type} Set Accuracy:", accuracy / validation_total * 100)
    with open(f"{directory_path}/{txt_file_path}.txt", 'a') as file:
        file.write(f"{set_type} Set Accuracy: {accuracy / validation_total * 100}\n")

    if word_cloud:
        create_word_cloud(word_cloud_sentiment_map, set_type, txt_file_path)
    return (accuracy / validation_total * 100), confusion_matrix

def part_b(file_path, algorithm_accuracy):
    
    df = pd.read_csv(file_path)
    random_accuracy = 0
    positive_accuracy = 0
    total = len(df)
    random_confusion_matrix = {sentiment: {sentiment: 0 for sentiment in SENTIMENTS} for sentiment in SENTIMENTS}
    positive_confusion_matrix = {sentiment: {sentiment: 0 for sentiment in SENTIMENTS} for sentiment in SENTIMENTS}

    for index, row in df.iterrows():
        sentiment = row['Sentiment']

        # random prediction 
        random_sentiment = SENTIMENTS[random.randint(0, 2)]
        if sentiment == random_sentiment:
            random_accuracy += 1
        random_confusion_matrix[random_sentiment][sentiment] += 1

        # positive prediction
        if sentiment == "Positive":
            positive_accuracy += 1
        positive_confusion_matrix["Positive"][sentiment] += 1

    print("Random Accuracy:", random_accuracy / total * 100)
    print("Improvement over Random:", algorithm_accuracy - (random_accuracy / total * 100))

    print("Positive Accuracy:", positive_accuracy / total * 100)
    print("Improvement over Positive:", algorithm_accuracy - (positive_accuracy / total * 100))

    with open(f"output/q1/b.txt", 'a') as file:
        file.write(f"Random Accuracy: {random_accuracy / total * 100}\n")
        file.write(f"Improvement over Random: {algorithm_accuracy - (random_accuracy / total * 100)}\n")
        file.write(f"Positive Accuracy: {positive_accuracy / total * 100}\n")
        file.write(f"Improvement over Positive: {algorithm_accuracy - (positive_accuracy / total * 100)}\n")

    return random_confusion_matrix, positive_confusion_matrix

def part_c(confusion_matrix, confusion_matrix_label):

    # Convert the confusion matrix to a 2D array
    confusion_matrix_array = np.array([[confusion_matrix['Positive']['Positive'], confusion_matrix['Positive']['Neutral'], confusion_matrix['Positive']['Negative']],
                                    [confusion_matrix['Neutral']['Positive'], confusion_matrix['Neutral']['Neutral'], confusion_matrix['Neutral']['Negative']],
                                    [confusion_matrix['Negative']['Positive'], confusion_matrix['Negative']['Neutral'], confusion_matrix['Negative']['Negative']]])

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  
    sns.heatmap(confusion_matrix_array, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Neutral', 'Negative'], yticklabels=['Positive', 'Neutral', 'Negative'])
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion Matrix')
    if not os.path.exists(f"{directory_path}/c/"):
        os.makedirs(f"{directory_path}/c/")
    plt.savefig(f"{directory_path}/c/{confusion_matrix_label}_confusion_matrix.png", dpi = 1000)
    plt.close()

def part_d_perform_lemmatizing(word):
    lemmatized_word = lemmatizer.lemmatize(word).lower()
    return lemmatized_word

if __name__ == '__main__':

    tweets_training_file_path = "Data/Corona_train.csv"
    tweets_validation_file_path = "Data/Corona_validation.csv"  

    # part a
    print(50*'-',"Part A", 50*'-')
    with open(f"{directory_path}/a.txt", 'w') as file:
        pass
    training_alogrithm_accuracy, training_confusion_matrix = part_a(tweets_training_file_path, tweets_training_file_path, "Training", "a", True)
    validation_alogrithm_accuracy, validation_confusion_matrix = part_a(tweets_training_file_path, tweets_validation_file_path, "validation", "a", True)

    # part b
    print(50*'-',"Part B", 50*'-')
    with open(f"{directory_path}/b.txt", 'w') as file:
        file.write("1.Training Set\n")
    print("1.Training Set")
    training_random_confusion_matrix, training_positive_confusion_matrix = part_b(tweets_training_file_path, training_alogrithm_accuracy)

    with open(f"{directory_path}/b.txt", 'a') as file:
        file.write("2.Validation Set\n")
    print("2.Validation Set")
    validation_random_confusion_matrix, validation_positive_confusion_matrix = part_b(tweets_validation_file_path, validation_alogrithm_accuracy)

    # part c 
    part_c(training_confusion_matrix, "training_alogrithm")
    part_c(validation_confusion_matrix, "validation_alogrithm")

    part_c(training_random_confusion_matrix, "training_random")
    part_c(training_positive_confusion_matrix, "training_positive")

    part_c(validation_random_confusion_matrix, "validation_random")
    part_c(validation_positive_confusion_matrix, "validation_positive")

    # part d
    print(50*'-',"Part D", 50*'-')
    with open(f"{directory_path}/d.txt", 'w') as file:
        pass
    training_alogrithm_lemmatizing_accuracy, training_confusion_lemmatizing_matrix = part_a(tweets_training_file_path, tweets_training_file_path, "Training", "d", True, True)
    validation_alogrithm_lemmatizing_accuracy, validation_confusion_lemmatizing_matrix = part_a(tweets_training_file_path, tweets_validation_file_path, "validation", "d", True, True)  

    # part f
    print(50*'-',"Part F", 50*'-')
    with open(f"{directory_path}/f.txt", 'w') as file:
        pass
    target_validation = "data/q1/Domain_Adaptation/Twitter_validation.csv"  
    target_domain_size = [1, 2, 5, 10, 25, 50, 100]
    domain_adaptation_accuracy_array = []
    domain_adaptation_without_source_accuracy_array = []
    for size in target_domain_size:
        target_domain = f"data/q1/Domain_Adaptation/Twitter_train_{size}.csv"  

        domain_adaptation_accuracy, domain_adaptation_matrix = part_a(tweets_training_file_path, target_validation, f"domain_adaptation {size}", "f",False, True, target_domain)  
        domain_adaptation_accuracy_array.append(domain_adaptation_accuracy)

        domain_adaptation_without_source_accuracy, domain_adaptation_without_source_matrix = part_a(target_domain, target_validation, f"domain_adaptation {size} without source", "f", False, True)  
        domain_adaptation_without_source_accuracy_array.append(domain_adaptation_without_source_accuracy)

    fig, ax = plt.subplots()

    ax.plot(target_domain_size, domain_adaptation_accuracy_array, label='With Domain Adaptation')
    ax.plot(target_domain_size, domain_adaptation_without_source_accuracy_array, label='Without Domain Adaptation')

    ax.set_xlabel('Target Subset size')
    ax.set_ylabel('Accuracy')
    ax.set_title('With Domain Adaptation vs. Without Domain Adaptation')
    ax.legend()

    plt.savefig(f"{directory_path}/f_accuracy_plot.png", dpi = quality)
    plt.close()