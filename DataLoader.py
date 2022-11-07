import re
import random
import warnings
import copy
import itertools

def load_train_data(train_input_path, train_truth_path, original=False):
    """
    Input:
    input_path: path containing training input data
    truth_path: path containing training truth data

    Output:
    Nested list of training data. Each sublist refers to an input read from train_input_path. Each item in sublist is of the form [sentence, 0/1, index of wrong word, correct word, wrong word], where 0 indicates wrong sentence and 1 indicates correct sentence. Sublists ensure that the right and wrong sentences of the same input are binded together.

    """
    train_input = open(train_input_path, encoding='utf-8', errors='ignore').readlines()
    train_true = open(train_truth_path, encoding='utf-8', errors='ignore').readlines()

    train_data = []

    for i in range(len(train_input)):
        inp = train_input[i].strip().split("\t")
        trth = train_true[i].strip().split(", ")
        sentence_data = []

        # For each input, get the right sentence
        sentence = inp[1]
        index_wrong_word = copy.deepcopy(trth)
        for j in range(1, len(trth), 2):  # iterate through the index of wrong word
            index = int(trth[j])
            right_word = trth[j + 1]
            wrong_word = sentence[index - 1]
            index_wrong_word[j + 1] = wrong_word
            sentence = sentence[:index - 1] + right_word + sentence[index:]

        right_sentence = sentence
        
        for k in range(1, len(trth), 2): 
            index = int(trth[k])
            right_word = trth[k + 1]
            wrong_word = index_wrong_word[k + 1]
            temp_wrong_sentences = right_sentence[:index - 1] + wrong_word + right_sentence[index:]
            
            # For each input sentence, bind right and wrong sentences together and store in sentence_data
            temp_list_right = []
            temp_list_right.append(right_sentence)
            temp_list_right.append(0) # here 0 indicates right sentences
            if not original:
                temp_list_right.append(index)
                temp_list_right.append(right_word)
                temp_list_right.append(wrong_word)
            if temp_list_right not in sentence_data:
                sentence_data.append(temp_list_right)

            temp_list_wrong = []
            temp_list_wrong.append(temp_wrong_sentences)
            temp_list_wrong.append(1) # here 1 indicates wrong sentences
            temp_list_wrong.append(index)
            temp_list_wrong.append(right_word)
            temp_list_wrong.append(wrong_word)
            sentence_data.append(temp_list_wrong)
            
        if sentence_data not in  train_data:
            train_data.append(sentence_data)

    return train_data

def shuffle(data, type):
    """
    Input:
    type: shuffling type 'pair' or 'single'
    data: training data to be shuffled

    Output:
    Shuffled data

    """
    if type == 'single' or type == 'original':
        flat_data = list(itertools.chain(*data))
        random.seed()
        random.shuffle(flat_data)
        print('Conducted {} shuffle'.format(type))
        return flat_data
    elif type == 'pair':
        random.seed()
        random.shuffle(data)
        flat_data = list(itertools.chain(*data))
        print('Conducted pair shuffle')
        return flat_data
    else:
        print('Shuffling type not accepted. Choose from ''single'' or ''pair''.')


def load_test_data(test_input_path, test_truth_path):
    """
    Input:
    input_path: path containing test input data
    truth_path: path containing test truth data

    Output:
    List of test data. Each item is of the form [sentence, 0/1, [index of wrong word 1, correct word 1, index of wrong word 2, correct word 2, ...]], where 0 indicates wrong sentence and 1 indicates correct sentence. If the sentence is correct, the third position of the item would be ['0'].

    """
    test_input = open(test_input_path, encoding='utf-8', errors='ignore').readlines()
    test_true = open(test_truth_path, encoding='utf-8', errors='ignore').readlines()

    test_data = []
    for i in range(len(test_input)):
        temp_list = []
        inp = test_input[i].strip().split("\t")
        trth = test_true[i].strip().split(", ")
        sentence = inp[1]
        if len(trth) == 2:
            flag = 0 # indication of right sentence
        else:
            flag = 1 # indication of wrong sentence
        right_word_info = trth[1:len(trth)]
        temp_list.append(sentence)
        temp_list.append(flag)
        temp_list.append(right_word_info)
        test_data.append(temp_list)

    return test_data


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    train_input_path = 'SICHAN13-14-15/TrainingInputAll.txt'
    train_truth_path = 'SICHAN13-14-15/TrainingTruthAll.txt'
    test_input_path = 'SICHAN13-14-15/TestInput.txt'
    test_truth_path = 'SICHAN13-14-15/TestTruth.txt'
    train_data = load_train_data(train_input_path, train_truth_path)
    shuffled_data = shuffle(train_data, 'pair')
    test_data = load_test_data(test_input_path, test_truth_path)