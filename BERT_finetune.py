from DataLoader import load_train_data, shuffle, load_test_data
from pandas.core.frame import DataFrame
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
import os
import argparse
from tqdm import tqdm
import random
import math
from english_dataset_preprocess import get_english_dataset


def load_dataset(train_input_path, train_truth_path, test_input_path, test_truth_path, shuffle_type, train_percentage = 1.0):
    """
    Input:
    train_input_path: path containing training input data
    train_truth_path: path containing training truth data
    test_input_path: path containing test input data
    test_truth_path: path containing test truth data
    shuffle_type: "single" or "pair"
    train_percentage: the percentage of training inputs to train the model

    Output:
    Two lists, one list contains train inputs and targets, and another list contains test inputs and targets

    """
    original = False
    if shuffle_type == 'original': original = True
    train_data = load_train_data(train_input_path, train_truth_path, original)
    shuffled_train_data = shuffle(train_data, shuffle_type)
    test_data = load_test_data(test_input_path, test_truth_path)
    
    end_select = math.ceil(train_percentage*len(shuffled_train_data))
    train, test = DataFrame(shuffled_train_data[0:end_select]), DataFrame(test_data)
    
    if shuffle_type == "pair" and end_select % 2 != 0:
        train = DataFrame(shuffled_train_data[0:end_select+1])
    
    train_inputs, train_targets = train[0].values, train[1].values
    test_inputs, test_targets = test[0].values, test[1].values
    
    print('Number of training inputs: ', len(train_inputs))
    print('Number of testing sentences: ', len(test_inputs))
    
    return [list(train_inputs), list(train_targets)], [list(test_inputs), list(test_targets)]


class BertClassificationModel(nn.Module):
    """
    BERT Classifier model: Use BERT_BASE model and apply a 768 * 2 fully connected layer to build the classifier
    
    """
    def __init__(self, model_name, device):
        super(BertClassificationModel, self).__init__()
        bert_config = BertConfig.from_pretrained(model_name)
        model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, model_name)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.encoder = BertEncoder(bert_config)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.embeddings = BertEmbeddings(bert_config)
        self.dense = nn.Linear(768, 2)  
        self.dropout = nn.Dropout(p=0.5)  
        self.device = device
        self.model_name = model_name

    def forward(self, batch_sentences):

        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, pad_to_max_length=True)  
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :] 
        self.mask_embeddings = self.embeddings.word_embeddings.weight[103]
        dropout_output = self.dropout(bert_cls_hidden_state)
        linear_output = self.dense(dropout_output)
    
        
        return linear_output
    
def train_model(model, criterion, optimizer, epochs, batch_size, device, train_data, test_data, shuffle_type):
    train_inputs = train_data[0]
    train_targets = train_data[1]
    test_inputs = test_data[0]
    test_targets = test_data[1]
    
    
    batch_count = int(len(train_inputs) / batch_size)
    batch_train_inputs, batch_train_targets = [], []
    for i in range(batch_count):
        batch_train_inputs.append(list(train_inputs[i * batch_size: (i + 1) * batch_size]))
        batch_train_targets.append(list(train_targets[i * batch_size: (i + 1) * batch_size]))
        
        
    evaluation_list = []
    train_loss = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for i in tqdm(range(batch_count)):
            inputs = batch_train_inputs[i]
            labels = torch.tensor(batch_train_targets[i]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss/batch_count)
        print("Epoch {}; Average train loss: {}".format(epoch, total_loss/batch_count))

        # Save models per epoch
        model.eval()
        # save_path = '{}/saved_models/BERT_{}/BERT_{}_epoch{}.pkl'.format(os.getcwd(), shuffle_type, shuffle_type, str(epoch))
        save_path = '{}/saved_models/BERT_english/BERT_english_epoch{}.pkl'.format(os.getcwd(), str(epoch))
        torch.save(model, save_path)
        acc_test, TP_test, FP_test, TN_test, FN_test = evaluate_model(model, test_inputs, test_targets)
        evaluation_list.append([acc_test, TP_test, FP_test, TN_test, FN_test])
        
    return evaluation_list, train_loss


def evaluate_model(model, inputs, targets):
    """
    Input:
    model: the model to be evaluated
    inputs: a list of inputs (usually test inputs)
    targets: a list contains all true labels (usually test targets)

    Output:
    acc: accuracy of the model
    TP: number of true positive
    FP: number of false positive
    TN: number of true negative
    FN: number of false negative
    cla_index: nested lists contain all indecies of those confusion matrix measures

    """  
    predictions = []
    with torch.no_grad():
        for i in range(len(inputs)):
            outputs = model([inputs[i]])
            predicted = torch.max(outputs, 1)
            predictions.append(int(predicted.indices))
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(targets)):
        if targets[i] == 1 and predictions[i] == 1: TP += 1
        if targets[i] == 0 and predictions[i] == 1: FP += 1
        if targets[i] == 0 and predictions[i] == 0: TN += 1
        if targets[i] == 1 and predictions[i] == 0: FN += 1
    
    acc = (TP + TN) / (TP + FP + TN + FN)
    print("Accuracy:", acc)
    return acc, TP, FP, TN, FN

def print_evaluation_data(model_path, test_data):
    test_inputs = test_data[0]
    test_targets = test_data[1]
    bert_classifier_model = torch.load(model_path).to(device)
    print("\nLabels: Wrong sentence (1); True sentence (0)")
    acc, TP, FP, TN, FN = evaluate_model(bert_classifier_model, test_inputs, test_targets)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", 2*precision*recall/(precision+recall))
    
    print("\nLabels: Wrong sentence (0); True sentence (1)")
    precision = TN/(TN+FN)
    recall = TN/(TN+FP)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", 2*precision*recall/(precision+recall))
    

def _parse_args():
    parser = argparse.ArgumentParser(description = 'Parameters')
    parser.add_argument('--train_input_path', type = str, default = 'SICHAN13-14-15/TrainingInputAll.txt', help = 'path to train data input file')
    parser.add_argument('--train_truth_path', type = str, default = 'SICHAN13-14-15/TrainingTruthAll.txt', help = 'path to train data target file')
    parser.add_argument('--test_input_path', type = str, default = 'SICHAN13-14-15/TestInput.txt', help = 'path to test data input file')
    parser.add_argument('--test_truth_path', type = str, default = 'SICHAN13-14-15/TestTruth.txt', help = 'path to test data target file')
    parser.add_argument('--epoch_num', type = int, default = 20, help = 'number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'number of batch size')
    parser.add_argument('--shuffle_type', type = str, default = 'pair', help = 'type of shuffle (single or pair or original)')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = _parse_args()
    
    # train_data, test_data = load_dataset(args.train_input_path, args.train_truth_path, args.test_input_path, args.test_truth_path, args.shuffle_type)
    train_data, test_data = get_english_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # bert_classifier_model = BertClassificationModel('bert-base-chinese', device).to(device)

    
    bert_classifier_model = BertClassificationModel('bert-base-uncased', device).to(device)

    params = bert_classifier_model.parameters()
    optimizer = torch.optim.Adam(params,
                                lr=2e-6,
                                betas=(0.9, 0.999),
                                eps=1e-8,
                                amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    
    evaluation_list, train_loss = train_model(bert_classifier_model, criterion, optimizer, args.epoch_num, args.batch_size, device, train_data, test_data, args.shuffle_type)
    acc_list = [epoch[0] for epoch in evaluation_list]
    max_acc_epoch = acc_list.index(max(acc_list))
    # max_saved_path = '{}/saved_models/BERT_{}/BERT_{}_epoch{}.pkl'.format(os.getcwd(), args.shuffle_type, args.shuffle_type, str(max_acc_epoch))
    max_saved_path = '{}/saved_models/BERT_english/BERT_english_epoch{}.pkl'.format(os.getcwd(), str(max_acc_epoch))
    print_evaluation_data(max_saved_path, test_data)
