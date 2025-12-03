# -*- coding: utf-8 -*-
import torch
import os
import csv
import random
import re
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.model_selection import train_test_split
import optuna # 記得 pip install optuna

# !pip install -U gdown -q
# !gdown --folder https://drive.google.com/drive/folders/1786AXJRAtqFvWMBeh-bLm4MtU21IQpBg
# !pip install gensim
# Training Config
DEVICE_NUM = 2
BATCH_SIZE = 64
EPOCH_NUM = 50
MAX_POSITIONS_LEN = 500
SEED = 2025
MODEL_DIR = 'model.pth'
lr = 0.001

# Set Seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RNN Config
w2v_config = {'path': 'model_3', 'dim': 256}

lstm_config = {
    'hidden_dim': 128,
    'num_layers': 1,        # 有了 Pack 和 Attention，1 層通常就夠強且夠快
    'bidirectional': True,
    'fix_embedding': False  # 建議解凍，因為我們有 Unlabeled Data 強化過 Word2Vec
}

header_config = {
    'dropout': 0.2,         # 因為解凍了 Embedding，Dropout 可以高一點防止過擬合
    'hidden_dim': 256       # 256 * 2
}

# 確保維度對應正確 (這行原本的 assert 很好，留著)
assert header_config['hidden_dim'] == lstm_config['hidden_dim'] or header_config['hidden_dim'] == lstm_config['hidden_dim'] * 2


"""## Utils & Classes (保持你的原本設計)"""

# Regex & Parsing
REGEX_URL = re.compile(r'http\S+')
REGEX_USER = re.compile(r'@\w+')
REGEX_HASHTAG = re.compile(r'#')
REGEX_REPEAT = re.compile(r'(.)\1{2,}')
REGEX_PUNCTUATION = re.compile(r'([^\w\s])')
REGEX_SPACES = re.compile(r'\s+')

def parsing_text(text):
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower() 
    text = REGEX_URL.sub('', text)
    text = REGEX_USER.sub('', text)
    text = REGEX_HASHTAG.sub('', text)
    text = text.replace("n't", " n't").replace("'s", " 's").replace("'m", " 'm") \
               .replace("'re", " 're").replace("'ve", " 've").replace("'ll", " 'll").replace("'d", " 'd")
    text = REGEX_REPEAT.sub(r'\1\1', text)
    text = REGEX_PUNCTUATION.sub(r' \1 ', text)
    text = REGEX_SPACES.sub(' ', text).strip()
    return text

# Loading Functions
def load_train_label(path='dataset/train_label.csv'):
    tra_lb_pd = pd.read_csv(path)
    idx = tra_lb_pd['id'].tolist()
    text = [parsing_text(s).split() for s in tra_lb_pd['text'].tolist()]
    label = tra_lb_pd['label'].tolist()
    return idx, text, label

def load_train_nolabel(path='dataset/train_nolabel.csv'):
    tra_nlb_pd = pd.read_csv(path)
    text = [parsing_text(s).split() for s in tra_nlb_pd['text'].tolist()]
    return text

def load_test(path='dataset/test.csv'):
    test_pd = pd.read_csv(path)
    idx = test_pd['id'].tolist()
    text = [parsing_text(s).split() for s in test_pd['text'].tolist()]
    return idx, text

# Iterators & Preprocessor
class SentenceIterator:
    def __init__(self, labeled_data, unlabeled_path):
        self.labeled_data = labeled_data
        self.unlabeled_path = unlabeled_path
    def __iter__(self):
        for sentence in self.labeled_data:
            yield sentence
        import csv
        with open(self.unlabeled_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) > 1:
                    text = parsing_text(row[1])
                    yield text.split()

class Preprocessor:
    def __init__(self, sentence_iterator, w2v_config):
        self.word2idx = {}
        self.idx2word = []
        self.embedding_matrix = []
        print("Training Bigram detector...")
        phrases = Phrases(sentence_iterator, min_count=5, threshold=10)
        self.bigram_transformer = Phraser(phrases)
        
        class BigramIterator:
            def __init__(self, iterator, transformer):
                self.iterator = iterator
                self.transformer = transformer
            def __iter__(self):
                for sentence in self.iterator:
                    yield self.transformer[sentence]
        bigram_sentences = BigramIterator(sentence_iterator, self.bigram_transformer)
        self.build_word2vec(bigram_sentences, **w2v_config)

    def build_word2vec(self, x, path, dim):
        if os.path.isfile(path):
            print("loading word2vec model ...")
            w2v_model = Word2Vec.load(path)
        else:
            print("training word2vec model ...")
            w2v_model = Word2Vec(x, vector_size=dim, window=5, min_count=2, workers=12, epochs=2, sg=1)
            print("saving word2vec model ...")
            w2v_model.save(path)
        self.embedding_dim = w2v_model.vector_size
        for i, word in enumerate(w2v_model.wv.key_to_index):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(w2v_model.wv[word])
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        print("total words: {}".format(len(self.embedding_matrix)))

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def sentence2idx(self, sentence):
        sentence = self.bigram_transformer[sentence] 
        sentence_idx = []
        for word in sentence:
            if word in self.word2idx.keys():
                sentence_idx.append(self.word2idx[word])
            else:
                sentence_idx.append(self.word2idx["<UNK>"])
        return torch.LongTensor(sentence_idx)

# Dataset
class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, id_list, sentences, labels, preprocessor):
        self.data = []
        self.sentences = sentences
        self.labels = labels # Remove to save memory if not needed
        self.id_list = id_list
        self.preprocessor = preprocessor
        MIN_LENGTH = 3 
        
        print("Preprocessing dataset...")
        for i in range(len(sentences)):
            input_ids = preprocessor.sentence2idx(sentences[i])
            if len(input_ids) >= MIN_LENGTH:
                label = labels[i] if labels is not None else None
                self.data.append({
                    'id': id_list[i],
                    'text': input_ids,
                    'label': label
                })
        print(f"Filtered {len(sentences) - len(self.data)} short/empty samples.")

    def __getitem__(self, idx):
        item = self.data[idx]
        if item['label'] is None:
            return item['id'], item['text']
        return item['id'], item['text'], item['label']

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data):
        id_list = torch.LongTensor([d[0] for d in data])
        lengths = torch.LongTensor([len(d[1]) for d in data])
        texts = pad_sequence([d[1] for d in data], batch_first=True, padding_value=0).contiguous()
        if len(data[0]) == 2:
            return id_list, lengths, texts
        labels = torch.FloatTensor([d[2] for d in data])
        return id_list, lengths, texts, labels

"""## Models (Backbone & Header)"""

class LSTM_Backbone(torch.nn.Module):
    def __init__(self, embedding, hidden_dim, num_layers, bidirectional, fix_embedding=False):
        super(LSTM_Backbone, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dropout = torch.nn.Dropout2d(0.3) 
        self.lstm = torch.nn.LSTM(embedding.size(1), hidden_dim, num_layers=num_layers, \
                                  bidirectional=bidirectional, batch_first=True)

    def forward(self, inputs, lengths):
        embeds = self.embedding(inputs)
        if self.training:
            embeds = embeds.permute(0, 2, 1)  
            embeds = embeds.unsqueeze(3)
            embeds = self.embedding_dropout(embeds)
            embeds = embeds.squeeze(3).permute(0, 2, 1)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return out

class Header(torch.nn.Module):
    def __init__(self, dropout, hidden_dim, num_heads=4):
        super(Header, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.attention_linear = torch.nn.Linear(hidden_dim, num_heads)
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, inputs, lengths):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        scores = self.attention_linear(inputs)
        mask = torch.arange(seq_len, device=inputs.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1).expand(-1, -1, self.num_heads)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)
        inputs_reshaped = inputs.view(batch_size, seq_len, self.num_heads, self.head_dim)
        weights_expanded = weights.unsqueeze(-1)
        context_vectors = torch.sum(inputs_reshaped * weights_expanded, dim=1)
        context_vectors = context_vectors.view(batch_size, -1)
        out = self.classifier(context_vectors).squeeze()
        return out

"""## Training Logic"""

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss

def train(train_loader, backbone, header, optimizer, criterion, device, epoch):
    alpha = 4.0 # R-Drop weight
    for i, (idx_list, lengths, texts, labels) in enumerate(train_loader):
        lengths, inputs, labels = lengths.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # R-Drop Logic
        inputs_doubled = torch.cat([inputs, inputs], dim=0)
        lengths_doubled = torch.cat([lengths, lengths], dim=0)
        if backbone is not None:
            feats = backbone(inputs_doubled, lengths_doubled)
        logits = header(feats, lengths_doubled)
        
        batch_size = inputs.size(0)
        logits1, logits2 = logits[:batch_size], logits[batch_size:]
        soft_predicted = (logits1 + logits2) / 2
        
        loss_nll = (criterion(logits1, labels) + criterion(logits2, labels)) / 2
        loss_kl = compute_kl_loss(logits1, logits2)
        loss = loss_nll + alpha * loss_kl
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            hard_predicted = (soft_predicted >= 0).int() 
            correct = sum(hard_predicted == labels).item()
            batch_size = len(labels)
            print('[ Epoch {}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(epoch+1,i+1, len(train_loader), loss.item(), correct * 100 / batch_size), end='\r')

def valid(valid_loader, backbone, header, criterion, device, epoch):
    backbone.eval()
    header.eval()
    with torch.no_grad():
        total_loss = []
        total_acc = []
        for i, (idx_list, lengths, texts, labels) in enumerate(valid_loader):
            lengths, inputs, labels = lengths.to(device), texts.to(device), labels.to(device)
            if not backbone is None:
                inputs = backbone(inputs, lengths)
            soft_predicted = header(inputs, lengths)
            loss = criterion(soft_predicted, labels)
            total_loss.append(loss.item())
            hard_predicted = (soft_predicted >= 0).int()
            correct = sum(hard_predicted.view(-1) == labels.view(-1)).item()
            acc = correct * 100 / len(labels)
            total_acc.append(acc)
    backbone.train()
    header.train()
    return np.mean(total_loss), np.mean(total_acc)

def run_training_final(train_loader, valid_loader, backbone, header, epoch_num, lr, weight_decay, device, model_dir):
    # 這是最後跑最佳參數用的，包含完整的 Log 和 Scheduler
    best_acc = 0.0
    patience = 5
    counter = 0
    optimizer = torch.optim.AdamW([{'params': backbone.parameters()}, {'params': header.parameters()}], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    backbone.train()
    header.train()
    
    for epoch in range(epoch_num):
        train(train_loader, backbone, header, optimizer, criterion, device, epoch)
        loss, acc = valid(valid_loader, backbone, header, criterion, device, epoch)
        print('[Validation in epoch {:}] loss:{:.3f} acc:{:.3f} '.format(epoch+1, loss, acc))
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save({'backbone': backbone, 'header': header}, model_dir)
            print(f'New best model saved! Acc: {best_acc:.3f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

def run_testing(test_loader, backbone, header, device, output_path):
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        with torch.no_grad():
            for i, (idx_list, lengths, texts) in enumerate(test_loader):
                lengths, inputs = lengths.to(device), texts.to(device)
                if not backbone is None:
                    inputs = backbone(inputs, lengths)
                soft_predicted = header(inputs, lengths)
                hard_predicted = (soft_predicted >= 0).int()
                for i, p in zip(idx_list, hard_predicted):
                    writer.writerow([str(i.item()), str(p.item())])

"""## Optuna Objective Function"""

def objective(trial):
    # 1. 建議參數
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)

    # 2. 建立 Loader (workers=0 避免卡死)
    train_loader_opt = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=0)
    valid_loader_opt = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn, num_workers=0)

    # 3. 建立模型
    lstm_config = {
        'hidden_dim': hidden_dim,
        'num_layers': 1,
        'bidirectional': True,
        'fix_embedding': False
    }
    # bidirectional 輸出是 hidden*2，所以 header 輸入是 hidden*2
    # hidden*2 一定能被 4 (num_heads) 整除 (128*2=256, 256*2=512)
    header_config = {
        'dropout': dropout,
        'hidden_dim': hidden_dim * 2,
        'num_heads': 4 
    }
    
    backbone = LSTM_Backbone(preprocessor.embedding_matrix, **lstm_config).to(device)
    header = Header(**header_config).to(device)

    optimizer = torch.optim.AdamW([
        {'params': backbone.parameters()},
        {'params': header.parameters()}
    ], lr=lr, weight_decay=weight_decay)
    
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4. 快速訓練 (只跑 10 epoch 來評估好壞)
    SEARCH_EPOCHS = 10
    best_acc = 0.0

    for epoch in range(SEARCH_EPOCHS):
        try:
            train(train_loader_opt, backbone, header, optimizer, criterion, device, epoch)
            loss, acc = valid(valid_loader_opt, backbone, header, criterion, device, epoch)
            
            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if acc > best_acc:
                best_acc = acc
        except Exception as e:
            # 避免梯度爆炸或其他錯誤導致整個 Optuna 停止
            print(f"Trial failed with error: {e}")
            raise optuna.TrialPruned()

    return best_acc

def get_pseudo_labels(dataset, backbone, header, device, threshold=0.9):
    # 建立一個循序的 Loader (Shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0)
    
    backbone.eval()
    header.eval()
    
    pseudo_data = []
    print(f"Generating pseudo labels with threshold {threshold}...")
    
    with torch.no_grad():
        for i, (idx_list, lengths, texts) in enumerate(loader):
            lengths, inputs = lengths.to(device), texts.to(device)
            
            if backbone is not None:
                inputs = backbone(inputs, lengths)
            soft_predicted = header(inputs, lengths)
            
            # 轉成機率 (因為用 BCEWithLogitsLoss，所以要加 Sigmoid)
            probs = torch.sigmoid(soft_predicted)
            probs = probs.cpu().tolist()
            
            # 根據 index 抓回原始句子
            # 假設 dataset.id_list 對應 dataset.sentences 的 index
            idx_list = idx_list.tolist()
            
            for j, prob in enumerate(probs):
                current_id = idx_list[j]
                # 這裡依賴 TwitterDataset 有保留 self.sentences
                raw_text = dataset.sentences[current_id] 
                
                if prob > threshold:
                    pseudo_data.append((raw_text, 1.0))
                elif prob < (1 - threshold):
                    pseudo_data.append((raw_text, 0.0))
                    
    print(f"Generated {len(pseudo_data)} pseudo-labeled samples.")
    return pseudo_data

"""## Main Execution"""
if __name__ == '__main__':
    # 環境設定
    SEED = 2025
    DEVICE_NUM = 2 # 你的設定
    EPOCH_NUM = 50
    MODEL_DIR = 'model.pth'
    w2v_config = {'path': 'model_3', 'dim': 256}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 準備資料
    print("Loading Data...")
    train_idx, train_label_text, label = load_train_label('dataset/train_label.csv')
    w2v_iterator = SentenceIterator(train_label_text, 'dataset/train_nolabel.csv')
    preprocessor = Preprocessor(w2v_iterator, w2v_config)

    train_idx, valid_idx, train_label_text, valid_label_text, train_label, valid_label = train_test_split(train_idx, train_label_text, label, test_size=0.12, random_state=SEED)
    
    print("Building Datasets...")
    train_dataset = TwitterDataset(train_idx, train_label_text, train_label, preprocessor)
    valid_dataset = TwitterDataset(valid_idx, valid_label_text, valid_label, preprocessor)

    # 2. 執行 Optuna 搜尋
    print("\n🚀 Starting Optuna Search...")
    # Pruner: 如果某次訓練在前幾個 epoch 比大多數 trials 都差，就直接砍掉
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    
    # n_trials=20: 嘗試 20 組參數
    study.optimize(objective, n_trials=20) 

    print("\n" + "="*50)
    print("🎉 Optimization Finished!")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("="*50)

    # 3. 使用最佳參數進行最終訓練
    print("\nTraining Final Model with Best Parameters...")
    best_params = study.best_params
    
    # 建立最佳 Loader
    final_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=0)
    final_valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=valid_dataset.collate_fn, num_workers=0)
    
    # 建立最佳 Model
    lstm_config = {
        'hidden_dim': best_params['hidden_dim'],
        'num_layers': 1,
        'bidirectional': True,
        'fix_embedding': False
    }
    header_config = {
        'dropout': best_params['dropout'],
        'hidden_dim': best_params['hidden_dim'] * 2,
        'num_heads': 4
    }
    
    backbone = LSTM_Backbone(preprocessor.embedding_matrix, **lstm_config).to(device)
    header = Header(**header_config).to(device)
    
    # 跑完整的 50 epoch
    run_training_final(final_train_loader, final_valid_loader, backbone, header, 50, best_params['lr'], best_params['weight_decay'], device, MODEL_DIR)

    print("\n🚀 Starting Self-Training (Pseudo-Labeling)...")
    
    # 3.1 載入 Unlabeled Data
    # 讀取文字檔
    train_nolabel_text = load_train_nolabel('dataset/train_nolabel.csv')
    # 建立 Dataset (只用來預測)
    unlabeled_dataset = TwitterDataset(range(len(train_nolabel_text)), train_nolabel_text, None, preprocessor)
    
    # 3.2 產生偽標籤
    # 使用剛剛練好的 Teacher (backbone, header) 去預測
    # threshold=0.9 代表非常有把握才收錄
    pseudo_samples = get_pseudo_labels(unlabeled_dataset, backbone, header, device, threshold=0.9)
    
    # 3.3 合併資料 (Original Labeled + Pseudo Labeled)
    print("Combining datasets...")
    # 提取偽標籤的文字和標籤
    pseudo_text = [p[0] for p in pseudo_samples]
    pseudo_label = [p[1] for p in pseudo_samples]
    
    # 合併
    student_train_text = train_label_text + pseudo_text
    student_train_label = train_label + pseudo_label
    student_train_idx = list(range(len(student_train_text)))
    
    # 3.4 建立 Student Dataset & Loader
    student_dataset = TwitterDataset(student_train_idx, student_train_text, student_train_label, preprocessor)
    student_loader = torch.utils.data.DataLoader(student_dataset, batch_size=best_params['batch_size'], shuffle=True, collate_fn=student_dataset.collate_fn, num_workers=0)
    
    # 3.5 初始化 Student Model (完全重練)
    print("Initializing Student Model...")
    student_backbone = LSTM_Backbone(preprocessor.embedding_matrix, **lstm_config).to(device)
    student_header = Header(**header_config).to(device)
    
    # 3.6 訓練 Student Model
    print("Training Student Model...")
    # 使用相同的最佳參數，但訓練資料變多了
    # 模型存檔名改為 'student_model.pth'
    run_training_final(student_loader, final_valid_loader, student_backbone, student_header, EPOCH_NUM, best_params['lr'], best_params['weight_decay'], device, 'student_model.pth')
    
    # =================================================================
    # [插入點] 結束
    # =================================================================

    # 4. 產生預測檔
    print("Generating Prediction...")
    test_idx, test_text = load_test('dataset/test.csv')
    test_dataset = TwitterDataset(test_idx, test_text, None, preprocessor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=0)
    
    run_testing(test_loader, backbone, header, device, 'pred_optuna.csv')
    print("Done! Check pred.csv")