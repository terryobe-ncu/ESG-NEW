import os.path

import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
HYPER_PARAMETERS = {
    'batch_size': 8,
    'lr': 1e-5,
    'epochs': 3,
}


class EsgBert(torch.nn.Module):
    def __init__(self):
        super(EsgBert, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('nbroad/ESG-BERT')

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output[0]
        return logits


class EsgDataset:
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        input_ids = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return input_ids

    def __len__(self):
        return len(self.encodings.input_ids)


class EsgBertPredict:
    def __init__(self, csv_path, output_dir, hyper_parameters=None) -> None:
        self.CSV_PATH = csv_path
        self.output_path = output_dir
        output_name = os.path.basename(csv_path).replace('.csv', '_label.csv')
        self.output_path = os.path.join(output_dir, output_name)

        self.HP = hyper_parameters or HYPER_PARAMETERS
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.df = None

    def _read_csv(self):
        self.df = pd.read_csv(self.CSV_PATH)
        x = self.df['paragraph']
        return x

    def _tokenizer(self, x):
        tokenizer = AutoTokenizer.from_pretrained('nbroad/ESG-BERT')
        inference_encodings = tokenizer(x.to_list(), truncation=True, padding=True)
        return inference_encodings

    def _get_dataset(self, inference_encodings):
        dataset = EsgDataset(inference_encodings)
        return dataset

    def _get_dataloader(self, inference) -> DataLoader:
        inference_loader = DataLoader(inference, batch_size=self.HP['batch_size'], shuffle=True)
        return inference_loader

    def _get_model(self) -> tuple[EsgBert, torch.optim.Optimizer]:
        model = EsgBert().to(DEVICE)
        optim = AdamW(model.parameters(), lr=1e-5)
        return model, optim

    @torch.no_grad()
    def inference(self, model: EsgBert, optim: torch.optim.Optimizer, inference_loader):
        count = 0
        model.eval()
        output_label = []
        loop = tqdm(inference_loader, leave=True)
        for batch_id, batch in enumerate(loop):
            optim.zero_grad()
            inputs = batch
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.argmax(outputs, dim=1)
            #  collect output into list
            outputs = outputs.cpu().tolist()
            output_label.extend(outputs)
            if batch_id % 50 == 0 and batch_id != 0:
                print(f'Epoch {batch_id}, count is {count}')
        self._gen_csv(output_label)

    def _gen_csv(self, output_label):
        self.df['label'] = output_label
        self.df.to_csv(self.output_path, index=False)

    def main(self):
        x = self._read_csv()
        inference_encodings = self._tokenizer(x)
        inference_dataset = self._get_dataset(inference_encodings)
        inference_loader = self._get_dataloader(inference_dataset)
        model, optim = self._get_model()
        self.inference(model, optim, inference_loader)
