import datetime
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


def load_bert_model(model_name_or_path='bert-base-uncased', save_path='../models/bert_models/', max_input_length=512,
                        num_labels=2):
    # initialize Bert configuration with specified max input length
    model_config = BertConfig(max_position_embeddings=max_input_length, num_labels=num_labels)

    # download pretrained model to specified directory
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                          cache_dir=save_path,
                                                          config=model_config)

    return model


def load_bert_tokenizer(model_name_or_path='bert-base-uncased', save_path='../models/bert_models/tokenizer', max_input_length=512,
                    num_labels=2):
    # initialize Bert configuration with specified max input length
    model_config = BertConfig(max_position_embeddings=max_input_length, num_labels=num_labels)

    # download model tokenizer to specified directory
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path, cache_dir=save_path,
                                              config=model_config)

    return tokenizer


def get_bert_embeddings(model, tokenizer, text=None):
    # use default sentence list if none is provided
    if text is None:
        text = ['This is a sample!', 'This is another longer sample text.']

    # tokenize sentences
    tokens_pt = tokenizer(text,  # text to be tokenized
                          padding=True,  # each sentence will have some PADDED tokens to match longest sequence length
                          truncation=False,  # truncate each sentence to the maximum length the model can accept (if applicable)
                          return_tensors="pt")  # get PyTorch style tensors

    # get token embeddings
    outputs = model(**tokens_pt)

    # get token embeddings
    last_hidden_state = outputs.last_hidden_state

    # get aggregated text representation
    pooler_output = outputs.pooler_output

    print("Token wise output shape: {}, Pooled output shape: {}".format(last_hidden_state.shape, pooler_output.shape))

    return pooler_output


def get_bert_encodings(tokenizer, texts_list):
    return tokenizer(texts_list, truncation=True, padding=True)


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_dataset(train_encodings, train_labels):
    return CustomDataset(train_encodings, train_labels)


def train_bert_model(model, train_dataset, output_dir='./bert_train_results',
                     result_model_dir='./bert_trained_model', logging_dir='./bert_train_logs'):
    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=300,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=logging_dir  # directory for storing logs
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset  # training dataset
    )

    trainer.train()
    trainer.save_model(result_model_dir)


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def eval_bert_model(model, tokenizer, test_dataset):
    eval_args = TrainingArguments(
        output_dir='./bert_eval_results',  # output directory
        per_device_eval_batch_size=64,  # batch size for evaluation
        logging_dir='./bert_eval_logs'  # directory for storing logs
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        tokenizer=tokenizer,  # tokenizer of the specified model
        args=eval_args,  # evaluation arguments, defined above
        eval_dataset=test_dataset  # evaluation dataset
    )

    trainer.evaluate()


if __name__ == '__main__':
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Downloading BERT model...')
    model = load_bert_model()
    tokenizer = load_bert_tokenizer()

    train_texts = ['This is a sample!', 'This is another longer sample text.']
    train_labels = [1, 0]

    test_texts = ['This is another sample!', 'This is yet another even longer sample text.']
    test_labels = [1, 0]

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Encoding input texts...')
    train_texts_encoded = get_bert_encodings(tokenizer, train_texts)
    test_texts_encoded = get_bert_encodings(tokenizer, test_texts)

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Generating datasets...')
    train_dataset = get_dataset(train_texts_encoded, train_labels)
    test_dataset = get_dataset(test_texts_encoded, test_labels)

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Training BERT model...')
    train_bert_model(model, train_dataset)

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Loading BERT model from trainer dir...')
    new_model = load_bert_model(model_name_or_path='./bert_trained_model')

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Evalutating BERT model...')
    eval_bert_model(new_model, tokenizer, test_dataset)
