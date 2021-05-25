import pandas as pd
from collections import defaultdict
import fasttext
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import pickle
from sklearn.utils import shuffle
from pprint import pprint
from scipy import stats

categories_values = {
    'gender': ['male', 'female', 'other'],
    'english_first_language': [0, 1],
    'age_group': ['Under 18', '18-30', '30-45', '45-60', 'Over 60', 'empty'],
    'logged_in': [True, False],
    'ns': ['article', 'user'],
    'sample': ['random', 'blocked'],
    'worker_id': range(5000),
    'year': range(2001, 2017),
    'education': ['none', 'some', 'hs', 'bachelors', 'masters', 'doctorate', 'professional']
}

age_group = {'Under 18', '18-30', '30-45', '45-60', 'Over 60', 'empty'}


def ddict():
    return defaultdict(ddict)


def prepare_data():
    print('Read data...')
    d = ddict()
    model = fasttext.load_model('../cc.en.300.bin')
    df = pd.read_csv('data/aggression/aggression_annotated_comments.tsv', sep='\t', encoding='utf-8')
    # comments
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        rev_id = row['rev_id']
        d['comments'][rev_id]['comment_orig'] = row['comment'].replace('NEWLINE_TOKEN', ' ')
        d['comments'][rev_id]['comment'] = model.get_sentence_vector(
            row['comment'].replace('NEWLINE_TOKEN', ' '))
        d['comments'][rev_id]['split'] = row['split']
        for col in ['year', 'logged_in', 'ns', 'sample']:
            d['comments'][rev_id][col] = label_binarize([row[col]], classes=categories_values[col])[0]
    # workers
    df = pd.read_csv('data/aggression/aggression_worker_demographics.tsv', sep='\t', encoding='utf-8')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        worker_id = row['worker_id']
        for col in ['gender', 'english_first_language', 'age_group', 'education']:
            if col == 'age_group' and row[col] not in age_group:
                row[col] = 'empty'
            d['workers'][worker_id][col] = label_binarize([row[col]], classes=categories_values[
                col])[0]
        d['workers'][worker_id]['worker_id'] = label_binarize([worker_id], classes=categories_values[
            'worker_id'])[0]
    X = dict()
    for s in ['s1', 's2', 's3', 's4']:
        X[s] = {'train': [], 'test': []}
    y = {'train': [], 'test': []}
    df = pd.read_csv('data/aggression/aggression_annotations.tsv', sep='\t', encoding='utf-8')
    w = defaultdict(lambda: defaultdict(int))
    outputs = set()
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        rev_id = row['rev_id']
        worker_id = row['worker_id']
        rev_type = 'test' if d['comments'][rev_id]['split'] == 'test' else 'train'
        if rev_type == 'train' and worker_id in d['workers']:
            output = row['aggression']
            outputs.add(output)
            worker_texts = d['workers'][worker_id]['texts']
            if not worker_texts[output]:
                worker_texts[output] = list()
            worker_texts[output].append(d['comments'][rev_id]['comment_orig'])
    for worker_id, worker_dict in tqdm(d['workers'].items()):
        for output in outputs:
            if not output in worker_dict['texts']:
                worker_dict['texts'][output] = {''}
            worker_dict['texts'][output] = model.get_sentence_vector(' '.join(worker_dict['texts'][output]))
    ann_ok = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        rev_id = row['rev_id']
        worker_id = row['worker_id']
        if worker_id not in d['workers']:  # or worker_id not in ok_workers:
            continue
        ann_ok += 1
        rev_type = 'test' if d['comments'][rev_id]['split'] == 'test' else 'train'
        X['s1'][rev_type].append(d['comments'][rev_id]['comment'])
        X['s2'][rev_type].append(np.hstack(
            [d['comments'][rev_id][x] for x in ['year', 'logged_in', 'ns', 'sample']] +
            [d['workers'][worker_id][x] for x in
             ['gender', 'english_first_language', 'age_group', 'education']]
        ))
        X['s3'][rev_type].append(d['workers'][worker_id]['worker_id'])
        X['s4'][rev_type].append(np.hstack([d['workers'][worker_id]['texts'][x] for x in [0, 1]]))
        output = row['aggression']
        y[rev_type].append(output)
    print('OK annotations: ', ann_ok)
    for rev_type in ['train', 'test']:
        np.save(f'input_f/s1_X_{rev_type}.npy', np.vstack(X['s1'][rev_type]).astype('float32'))
        np.save(f'input_f/s2_X_{rev_type}.npy', np.hstack((
            np.vstack(X['s1'][rev_type]).astype('float32'),
            np.vstack(X['s2'][rev_type]).astype('float32'))).astype('float32'))
        np.save(f'input_f/s3_X_{rev_type}.npy', np.hstack((
            np.vstack(X['s1'][rev_type]).astype('float32'),
            np.vstack(X['s2'][rev_type]).astype('float32'),
            np.vstack(X['s3'][rev_type]).astype('float32'))).astype('float32'))
        np.save(f'input_f/s4_X_{rev_type}.npy', np.hstack((
            np.vstack(X['s1'][rev_type]).astype('float32'),
            np.vstack(X['s2'][rev_type]).astype('float32'),
            np.vstack(X['s4'][rev_type]).astype('float32'))).astype('float32'))
        np.save(f'input_f/y_{rev_type}.npy', np.vstack(y[rev_type]).astype('float32'))


def evaluate():
    print("***EVALUATE***")
    results = dict()
    for s in ['s1', 's2']:#, 's3', 's4']:
        results[s] = dict()
        print(f'**Scenario: {s}')
        for i in range(10):
            print(f'****Iteration: {i}')
            X_train = np.load(f'input_f/{s}_X_train.npy') [:100000,:]
            X_test = np.load(f'input_f/{s}_X_test.npy')
            y_train = np.load('input_f/y_train.npy') [:100000,:]
            y_test = np.load('input_f/y_test.npy')
            X_train, y_train = shuffle(X_train, y_train)
            N = X_train.shape[0]
            Nfrac = int(N * 0.9)
            X_train, y_train = X_train[:Nfrac, :], y_train[:Nfrac, :]
            lr = LogisticRegression(max_iter=2000, verbose=1)
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, digits=4)
            print(classification_report(y_test, y_pred, digits=4))
            results[s][i] = report
            print()
    with open('results.dat', 'wb') as f:
        pickle.dump(results, f)


def stats_calc():
    alpha = 0.05
    samples = dict()
    with open('results.dat', 'rb') as f:
        results = pickle.load(f)
        for s in ['s1', 's2', 's3', 's4']:
            d = defaultdict(list)
            for i in range(10):
                r = results[s][i]
                d['accuracy'].append(r['accuracy'])
                d['p-macro'].append(r['macro avg']['precision'])
                d['r-macro'].append(r['macro avg']['recall'])
                d['f1-macro'].append(r['macro avg']['f1-score'])
                d['p-a'].append(r['1.0']['precision'])
                d['r-a'].append(r['1.0']['recall'])
                d['f1-a'].append(r['1.0']['f1-score'])
            samples[s] = d
            for measure in ['accuracy', 'p-macro', 'r-macro', 'f1-macro', 'p-a', 'r-a', 'f1-a']:
                stat, p = stats.shapiro(d[measure])
                #p = round(p, 6)
                if p > 0.05:
                    result05 = 'Gaussian'
                else:
                    result05 = 'Fail'
                if p > 0.01:
                    result01 = 'Gaussian'
                else:
                    result01 = 'Fail'
                #print(f'Set: {s},  Measure: {measure}, Shapiro-Wilk: p-val:{p}, for alpha=0.05: {result05}, for alpha=0.01: {result01}')
                row_format = "{:>25}" * 5
                print(row_format.format(s, measure, p, result05, result01))

    # for sa, sb in [('s1', 's2'), ('s1', 's3'), ('s1', 's4'), ('s2', 's3'), ('s2', 's4'), ('s3', 's4')]:
    for sa, sb in [('s1', 's2'), ('s1', 's3'), ('s1', 's4'), ('s2', 's3'),  ('s3', 's4')]:
        print(f'Pair: ({sa},{sb})')
        for measure in ['accuracy', 'p-macro', 'r-macro', 'f1-macro', 'p-a', 'r-a', 'f1-a']:
            stat, p = stats.ttest_rel(samples[sa][measure], samples[sb][measure])
            #p = round(p, 6)
            if p > 0.05:
                result05 = 'Fail'
            else:
                result05 = 'Significant'
            if p > 0.01:
                result01 = 'Fail'
            else:
                result01 = 'Significant'
            #print(f'Measure: {measure}, Paired sample t-test:  p-val: {p}, for alpha=0.05: {result05}, for alpha=0.01: {result01}')
            row_format = "{:>25}" * 6
            print(row_format.format(sa, sb, measure, p, result05, result01))

    #for s in ['s1', 's2', 's3', 's4']:
    #    for measure in ['accuracy', 'p-macro', 'r-macro', 'f1-macro', 'p-a', 'r-a', 'f1-a']:
    #        row_format = "{:>25}" * 4


# prepare_data()
# evaluate()
stats_calc()
