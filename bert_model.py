import torch
import pandas as pd
import os
import ast
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss, LogSoftmax, Linear, Sigmoid
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

HYPER_PARAMETERS = {
    'cuda_visible_devices': '0',
    'batch_size': 8,
    'lr': 1e-5,
    'epoch': 1,
    'threshold': 0.02
}
os.environ["CUDA_VISIBLE_DEVICES"] = HYPER_PARAMETERS['cuda_visible_devices']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class EsgBert(torch.nn.Module):
    def __init__(self):
        super(EsgBert, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('nbroad/ESG-BERT')
        self.fc = Linear(26, 27)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        out = self.fc(logits)
        output_layer = LogSoftmax(dim=1)
        return output_layer(out)


class TrainDataset:
    def __init__(self, encodings, y):
        self.encodings = encodings
        self.y = y

    def __getitem__(self, idx):
        input_ids = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label = torch.tensor(self.y[idx])
        return input_ids, label

    def __len__(self):
        return len(self.encodings.input_ids)


class TestDataset:
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        input_ids = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return input_ids

    def __len__(self):
        return len(self.encodings.input_ids)


class EsgBertPredict:
    def __init__(self, hyper_parameters=None):
        self.HP = hyper_parameters or HYPER_PARAMETERS
        self.device = DEVICE
        self.df_train = self.df_test = None
        self.train_csv = None
        self.model, self.optim, self.criterion = self._get_model()
        self.train_loader = self.valid_loader = None

    @staticmethod
    def _convert_onehot(row):
        y = []
        for i in range(27):
            if i in row:
                y.append(1)
            else:
                y.append(0)
        return y

    def _preprocess(self, df: pd.DataFrame):
        df['label'] = df['label'].apply(ast.literal_eval)
        df['label'] = df['label'].apply(self._convert_onehot)
        x_train = df['paragraph']
        y_train = df['label']
        return df, x_train, y_train

    @staticmethod
    def _tokenizer(x: pd.Series):
        tokenizer = AutoTokenizer.from_pretrained('nbroad/ESG-BERT')
        encodings = tokenizer(x.to_list(), truncation=True, padding=True)
        return encodings

    @staticmethod
    def _get_train_dataset(train_encodings, y_train):
        dataset = TrainDataset(train_encodings, y_train)
        train, valid = train_test_split(dataset, test_size=0.2, random_state=42)
        return train, valid

    def _get_dataloader(self, dataset) -> DataLoader:
        data_loader = DataLoader(dataset, batch_size=self.HP['batch_size'], shuffle=True)
        return data_loader

    @staticmethod
    def _get_model():
        model = EsgBert().to(DEVICE)
        optim = AdamW(model.parameters(), lr=1e-5)
        criterion = CrossEntropyLoss()
        return model, optim, criterion

    def train_data(self, train_csv):
        self.train_csv = train_csv
        df_train = pd.read_csv(self.train_csv, dtype={'label': object})
        self.df_train, x_train, y_train = self._preprocess(df_train)
        train_encodings = self._tokenizer(x_train)
        train_dataset, valid_dataset = self._get_train_dataset(train_encodings, y_train)
        self.train_loader, self.valid_loader = self._get_dataloader(train_dataset), self._get_dataloader(valid_dataset)

    def train(self):
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.train_loader, leave=True)
        for batch in loop:
            self.optim.zero_grad()
            inputs, y = batch
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            y = y.to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, y.float())
            # calculate loss
            loss.backward()
            # update parameters
            self.optim.step()
            running_loss += loss.item()
        return running_loss

    def training(self):
        for epoch in range(self.HP['epoch']):
            print(f'Epoch: {epoch} Train Loss {self.train():.4f}')
            print(f'Epoch: {epoch} Valid Loss {self.valid():.4f}')

    @torch.no_grad()
    def valid(self):
        self.model.eval()
        running_loss = 0.0
        loop = tqdm(self.valid_loader, leave=True)
        for batch in loop:
            self.optim.zero_grad()
            inputs, y = batch
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            y = y.to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, y.float())
            loss.detach().cpu()
            #  collect output into list
            outputs.cpu().tolist()
            running_loss += loss.item()
        return running_loss

    @torch.no_grad()
    def test(self, test_csv_path, test_output_dir):
        output_name = os.path.basename(test_csv_path).replace('.csv', '_label.csv')
        output_path = os.path.join(test_output_dir, output_name)
        self.df_test = pd.read_csv(test_csv_path)
        x_test = self.df_test['paragraph']
        test_encodings = self._tokenizer(x_test)
        test_dataset = TestDataset(test_encodings)
        test_loader = self._get_dataloader(test_dataset)

        self.model.eval()
        output_label = []
        loop = tqdm(test_loader, leave=True)
        sigmoid = Sigmoid()
        for batch in loop:
            self.optim.zero_grad()
            inputs = batch
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = sigmoid(outputs)
            # predicted_labels = (outputs > self.HP['threshold']).int()
            predicted_labels = outputs.cpu().tolist()
            output_label.extend(predicted_labels)
        self.df_test['label'] = output_label
        self.df_test['label'] = self.df_test['label'].apply(lambda x: [round(val, 3) for val in x])
        self.df_test.to_csv(output_path, index=False)
