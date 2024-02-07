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
    def __init__(self, train_csv_path, test_csv_path, test_csv_output, nation, train_source, test_source, test_output,
                 hyper_parameters) -> None:
        self.TRAIN_CSV_PATH = train_csv_path
        self.TEST_CSV_PATH = test_csv_path
        self.TEST_CSV_OUTPUT_PATH = test_csv_output
        self.NATION = nation
        self.TRAIN_SOURCE = train_source
        self.TEST_SOURCE = test_source
        self.TEST_OUTPUT = test_output
        self.HP = hyper_parameters
        self.device = DEVICE
        self.df_train = None
        self.df_test = None

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

    def _read_csv(self, path: str):
        self.df_train = pd.read_csv(path, dtype={'label': object})

    def _tokenizer(self, x: pd.Series):
        tokenizer = AutoTokenizer.from_pretrained('nbroad/ESG-BERT')
        encodings = tokenizer(x.to_list(), truncation=True, padding=True)
        return encodings

    def _get_train_dataset(self, train_encodings, y_train):
        dataset = TrainDataset(train_encodings, y_train)
        train, valid = train_test_split(dataset, test_size=0.2, random_state=42)
        return train, valid

    def _get_test_dataset(self, train_encodings):
        dataset = TestDataset(train_encodings)
        return dataset

    def _get_dataloader(self, dataset) -> DataLoader:
        data_loader = DataLoader(dataset, batch_size=self.HP['batch_size'], shuffle=True)
        return data_loader

    def _get_model(self):
        model = EsgBert().to(DEVICE)
        optim = AdamW(model.parameters(), lr=1e-5)
        criterion = CrossEntropyLoss()
        return model, optim, criterion

    def train(self, model: EsgBert, optim: torch.optim.Optimizer, criterion, train_loader, epoch):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for _, batch in enumerate(loop):
            optim.zero_grad()
            inputs, y = batch
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            y = y.to(self.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, y.float())
            # calculate loss
            loss.backward()
            # update parameters
            optim.step()
            running_loss += loss.item()
        print(f'Epoch: {epoch} Loss {running_loss:.4f}')
        running_loss = 0.0

    @torch.no_grad()
    def valid(self, model: EsgBert, optim: torch.optim.Optimizer, criterion, valid_loader, epoch):
        model.eval()
        running_loss = 0.0
        loop = tqdm(valid_loader, leave=True)
        for _, batch in enumerate(loop):
            optim.zero_grad()
            inputs, y = batch
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            y = y.to(self.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, y.float())
            loss.detach().cpu()
            #  collect output into list
            outputs = outputs.cpu().tolist()
            running_loss += loss.item()
        print(f'Epoch {epoch} Loss {running_loss:.4f}')
        running_loss = 0.0

    @torch.no_grad()
    def test(self, model: EsgBert, optim: torch.optim.Optimizer, test_loader):
        model.eval()
        output_label = []
        loop = tqdm(test_loader, leave=True)
        sigmoid = Sigmoid()
        for batch_id, batch in enumerate(loop):
            optim.zero_grad()
            inputs = batch
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = sigmoid(outputs)
            predicted_labels = (outputs > self.HP['threshold']).int()
            predicted_labels = predicted_labels.cpu().tolist()
            output_label.extend(predicted_labels)
        self._gen_csv(output_label)

    def _gen_csv(self, output_label):
        self.df_test['label'] = output_label
        self.df_test.to_csv(f'{self.TEST_CSV_OUTPUT_PATH}/{self.NATION}/{self.TEST_OUTPUT}.csv', index=False)

    def main(self):
        df_train = pd.read_csv(f'{self.TRAIN_CSV_PATH}/{self.NATION}/{self.TRAIN_SOURCE}.csv', dtype={'label': object})
        self.df_train, x_train, y_train = self._preprocess(df_train)
        train_encodings = self._tokenizer(x_train)
        train_dataset, valid_dataset = self._get_train_dataset(train_encodings, y_train)
        train_loader, valid_loader = self._get_dataloader(train_dataset), self._get_dataloader(valid_dataset)
        model, optim, criterion = self._get_model()
        # train & validate
        for epoch in range(self.HP['epoch']):
            self.train(model, optim, criterion, train_loader, epoch)
            self.valid(model, optim, criterion, valid_loader, epoch)

        # test
        self.df_test = pd.read_csv(f'{self.TEST_CSV_PATH}/{self.NATION}/{self.TEST_SOURCE}.csv')
        x_test = self.df_test['paragraph']
        test_encodings = self._tokenizer(x_test)
        test_dataset = self._get_test_dataset(test_encodings)
        test_loader = self._get_dataloader(test_dataset)
        self.test(model, optim, test_loader)
