from sklearn.model_selection import train_test_split
import numpy as np

def get_english_dataset():
    print("Preprocess English dataset...")
    corpus = open('corpus.data', encoding='utf-8', errors='ignore').readlines()
    all_sentences = corpus[0].split('<s>')[3:]

    errors = open('spellerrors.data', encoding='utf-8', errors='ignore').readlines()
    words_with_error = {}
    for words in errors:
        l = words.strip('\n').split(":")
        wrong_words = l[1].split(",")
        wrong_words = [w.strip(" ") for w in wrong_words]
        words_with_error[l[0]] = wrong_words
        
    right_wrong_sentences = []
    targets = []

    for sentence in all_sentences:
        if len(sentence) > 500: continue
        sentence_list = sentence.strip(" ").split(" ")
        for i in range(len(sentence_list)):
            word = sentence_list[i]
            if word in words_with_error:
                right_wrong_sentences.append(sentence.strip(" "))
                targets.append(0)
                wrong_words = words_with_error[word]
                sentence_list[i] = wrong_words[0]
                right_wrong_sentences.append(" ".join(sentence_list))
                targets.append(1)
                break
                    
    right_wrong_sentences = np.array(right_wrong_sentences)
    targets = np.array(targets)
    X_train, X_test, y_train, y_test = train_test_split(right_wrong_sentences, targets, test_size=0.3, random_state=42)
    print("Number of training sentences:", len(X_train))
    print("DONE!")
    return [X_train,y_train], [X_test, y_test]