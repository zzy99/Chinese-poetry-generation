import numpy as np
import json, codecs
from bert4keras.backend import keras, K
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import parallel_apply, sequence_padding
from bert4keras.snippets import DataGenerator

config_path = 'D:/Homework3/Data/bert/albert_small_zh_google/albert_config_small_google.json'
checkpoint_path = 'D:/Homework3/Data/bert/albert_small_zh_google/albert_model.ckpt'
dict_path = 'D:/Homework3/Data/bert/albert_small_zh_google/vocab.txt'

train, valid, test = [], [], []
text = codecs.open('train.txt', encoding='utf-8')
for line in text.readlines():
    line = line.strip().replace(',','').replace('.','').replace(' ','')
    train.append(line)
print(train[:10])

text = codecs.open('val.txt', encoding='utf-8')
for line in text.readlines():
    line = line.strip().replace(',','').replace('.','').replace(' ','')
    valid.append(line)
print(valid[:10])

text = codecs.open('test.txt', encoding='utf-8')
for line in text.readlines():
    line = line.strip().replace(',','').replace('.','').replace(' ','')
    test.append(line)
print(test[:10])

_token_dict = load_vocab(dict_path) 
_tokenizer = Tokenizer(_token_dict)  

tokens = json.load(open('seq2seq_config.json',encoding='utf-8'))
token_dict, keep_words = {}, [] 

for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
    token_dict[t] = len(token_dict)
    keep_words.append(_token_dict[t])

for t in tokens:
    if t in _token_dict and t not in token_dict:
        token_dict[t] = len(token_dict)
        keep_words.append(_token_dict[t])

tokenizer = Tokenizer(token_dict)

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids = [], []
        for i in idxs:
            sen = self.data[i]
            if len(sen) > 0:
                s1 = sen[:7]
                s234 = sen[7:28]
                token_ids, segment_ids = tokenizer.encode(s1, s234)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

model = build_bert_model(
        config_path,
        checkpoint_path,
        application='seq2seq',
        model='albert',
        keep_words=keep_words,
)

model.summary()

y_in = model.input[0][:, 1:] 
y_mask = model.input[1][:, 1:]
y_out = model.output[:, :-1]
cross_entropy = K.sparse_categorical_crossentropy(y_in, y_out)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam())

def predict(s, topk=10, l=21+1):
    token_ids, segment_ids = tokenizer.encode(s)
    target_ids = [[] for _ in range(topk)] 
    target_scores = [0] * topk 
    for i in range(l):
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model.predict([_target_ids, _segment_ids
                                 ])[:, -1, 3:] 
        _log_probas = np.log(_probas + 1e-6)
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:] 
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            if i == 0 and j > 0:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:]  
        target_ids = [_candidate_ids[k] for k in _topk_arg]
        target_scores = [_candidate_scores[k] for k in _topk_arg]
        best_one = np.argmax(target_scores)
        if target_ids[best_one][-1] == 3:
            return tokenizer.decode(target_ids[best_one])
    return tokenizer.decode(target_ids[np.argmax(target_scores)])

def show():
    s1 = u'七里虹桥腐草腥'
    s2 = u'黄衣舍去混缁衣'
    s3 = u'起从何处止何日'
    s4 = u'闭户跏趺意已清'
    s5 = u'夜寒金屋篆烟飞'
    for s in [s1, s2, s3, s4, s5]:
        a = predict(s)
        b = ' '.join([a[i:i + 7] for i in range(0, len(a), 7)])
        print(s, b)

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model_on_val.weights')
        show()

evaluator = Evaluate()
train_generator = data_generator(train, 512)
val_generator = data_generator(valid, 1)

model.load_weights('./best_model.weights')
# model.fit_generator(train_generator.forfit(),
#                     steps_per_epoch=10,
#                     epochs=100,
#                     validation_data=val_generator.forfit(),
#                     validation_steps=1000,
#                     callbacks=[evaluator]
#                     )

show()

with open('res.txt','w',encoding='utf-8')as f:
    for sen in test:
        ans = predict(sen,50)
        b = ' '.join([ans[i:i + 7] for i in range(0, len(ans), 7)])
        print(sen, b)
        f.write(b+'\n')