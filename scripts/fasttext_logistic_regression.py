import datetime
import json
import pickle as pkl
import random
import sys
import time
from collections import defaultdict

import fasttext
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from tqdm import tqdm


def generate_new_worker_embeddings(scenario_number='1', dev_texts=20, pure_text_embeddings=False):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Reading data...')

    # load fasttext model
    model = fasttext.load_model('../cc.en.300.bin')

    # read dataframe with all the data
    df = pd.read_csv('../data/merged_folds.csv', sep=',', encoding='utf-8')

    # read JSON containing ids of dev split reviews for each worker
    with open('../data/docs_for_embeddings.json', 'r') as f:
        dev_rev_id = json.load(f)

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
          f'Generating personal embeddings for scenario: {scenario_number} | dev_texts: {dev_texts}...')

    worker_embeddings = defaultdict()

    # most controversial, random class (0 and 1) distribution
    if scenario_number == '1':
        for i, worker_id in enumerate(df['worker_id'].unique()):
            # get controversial-sorted sample of rev_ids for personal embeddings
            worker_dev_rev_ids = dev_rev_id[str(worker_id)]['controversial'][:dev_texts]

            # get positive and negative texts lists
            worker_df = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.rev_id.isin(worker_dev_rev_ids))].reset_index(drop=True)

            single_text_representation_length = 305

            if pure_text_embeddings:
                single_text_representation_length = 300

            worker_embedding = np.empty((len(worker_df) * single_text_representation_length), dtype=np.float32)

            for index, row in worker_df.iterrows():
                text_embedding = model.get_sentence_vector(row.comment)
                text_entropy = row.entropy
                text_p_aggressive = row.p_aggressive
                text_aggression = row.aggression
                text_is_decision_major = row.is_decision_major
                text_overall_prc_mainstream = row.overall_prc_mainstream
                single_text_representation = np.hstack((text_embedding,
                                                        text_entropy,
                                                        text_p_aggressive,
                                                        text_aggression,
                                                        text_is_decision_major,
                                                        text_overall_prc_mainstream))

                if pure_text_embeddings:
                    single_text_representation = text_embedding

                worker_embedding[index * single_text_representation_length: (index+1) * single_text_representation_length] = single_text_representation

            worker_embeddings[worker_id] = worker_embedding

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        if pure_text_embeddings:
            dir_name = 'new_pure_worker_embeddings'
        else:
            dir_name = 'new_worker_embeddings'
        with open(f'{dir_name}/new_worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)

    # most aggressive, random class (0 and 1) distribution
    if scenario_number == '2':
        for i, worker_id in enumerate(df['worker_id'].unique()):
            # get positive and negative texts lists
            worker_df = df[(df.worker_id == worker_id) & (df.split == 'dev')]
            worker_df = worker_df.sort_values(by='p_aggressive', ascending=False)
            worker_df = worker_df[:dev_texts].reset_index(drop=True)

            single_text_representation_length = 305

            if pure_text_embeddings:
                single_text_representation_length = 300

            worker_embedding = np.empty((len(worker_df) * single_text_representation_length), dtype=np.float32)

            for index, row in worker_df.iterrows():
                text_embedding = model.get_sentence_vector(row.comment)
                text_entropy = row.entropy
                text_p_aggressive = row.p_aggressive
                text_aggression = row.aggression
                text_is_decision_major = row.is_decision_major
                text_overall_prc_mainstream = row.overall_prc_mainstream
                single_text_representation = np.hstack((text_embedding,
                                                        text_entropy,
                                                        text_p_aggressive,
                                                        text_aggression,
                                                        text_is_decision_major,
                                                        text_overall_prc_mainstream))

                if pure_text_embeddings:
                    single_text_representation = text_embedding
                worker_embedding[index * single_text_representation_length: (index+1) * single_text_representation_length] = single_text_representation

            worker_embeddings[worker_id] = worker_embedding

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        if pure_text_embeddings:
            dir_name = 'new_pure_worker_embeddings'
        else:
            dir_name = 'new_worker_embeddings'
        with open(f'{dir_name}/new_worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)

    # most controversial, equal class (0 and 1) distribution
    if scenario_number == '3':
        for i, worker_id in enumerate(df['worker_id'].unique()):

            if not pure_text_embeddings:
                # get positive and negative texts lists
                pos_worker_ids = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 0.0)]
                pos_worker_ids = pos_worker_ids.sort_values(by='entropy', ascending=False)

                if len(pos_worker_ids) >= (dev_texts // 2):
                    if dev_texts == 1:
                        pos_worker_ids = pos_worker_ids['rev_id'][:dev_texts].tolist()
                    else:
                        pos_worker_ids = pos_worker_ids['rev_id'][:dev_texts//2].tolist()
                else:
                    additional_pos_worker_ids = random.choices(pos_worker_ids['rev_id'].tolist(), k=(dev_texts // 2) - len(pos_worker_ids))
                    pos_worker_ids = pos_worker_ids['rev_id'].tolist() + additional_pos_worker_ids

                # get negative and negative texts lists
                neg_worker_ids = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 1.0)]
                neg_worker_ids = neg_worker_ids.sort_values(by='entropy', ascending=False)

                if len(neg_worker_ids) >= (dev_texts - len(pos_worker_ids)):
                    neg_worker_ids = neg_worker_ids['rev_id'][:dev_texts - len(pos_worker_ids)].tolist()
                else:
                    additional_neg_worker_ids = random.choices(neg_worker_ids['rev_id'].tolist(), k=(dev_texts - len(pos_worker_ids)) - len(neg_worker_ids))
                    neg_worker_ids = neg_worker_ids['rev_id'].tolist() + additional_neg_worker_ids

                worker_rev_ids = pos_worker_ids + neg_worker_ids

            else:
                # get positive and negative texts lists
                pos_worker_ids = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 0.0)]
                pos_worker_ids = pos_worker_ids.sort_values(by='entropy', ascending=False)

                if len(pos_worker_ids) >= (dev_texts // 2):
                    if dev_texts == 1:
                        pos_worker_ids = pos_worker_ids['rev_id'][:dev_texts].tolist()
                    pos_worker_ids = pos_worker_ids['rev_id'][:dev_texts//2].tolist()
                else:
                    additional_pos_worker_ids = random.choices(pos_worker_ids['rev_id'].tolist(), k=(dev_texts // 2) - len(pos_worker_ids))
                    pos_worker_ids = pos_worker_ids['rev_id'].tolist() + additional_pos_worker_ids

                # get negative and negative texts lists
                neg_worker_ids = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 1.0)]
                neg_worker_ids = neg_worker_ids.sort_values(by='entropy', ascending=False)

                if len(neg_worker_ids) >= (dev_texts - len(pos_worker_ids)):
                    neg_worker_ids = neg_worker_ids['rev_id'][:dev_texts - len(pos_worker_ids)].tolist()
                else:
                    additional_neg_worker_ids = random.choices(neg_worker_ids['rev_id'].tolist(), k=(dev_texts - len(pos_worker_ids)) - len(neg_worker_ids))
                    neg_worker_ids = neg_worker_ids['rev_id'].tolist() + additional_neg_worker_ids

                worker_rev_ids = pos_worker_ids + neg_worker_ids

            single_text_representation_length = 305

            if pure_text_embeddings:
                single_text_representation_length = 300

            worker_embedding = np.empty((len(worker_rev_ids) * single_text_representation_length), dtype=np.float32)

            for index, rev_id in enumerate(worker_rev_ids):
                annotation_df = df[(df.rev_id == rev_id) & (df.worker_id == worker_id)]
                text_embedding = model.get_sentence_vector(annotation_df.comment.tolist()[0])
                text_entropy = annotation_df.entropy.tolist()[0]
                text_p_aggressive = annotation_df.p_aggressive.tolist()[0]
                text_aggression = annotation_df.aggression.tolist()[0]
                text_is_decision_major = annotation_df.is_decision_major.tolist()[0]
                text_overall_prc_mainstream = annotation_df.overall_prc_mainstream.tolist()[0]

                single_text_representation = np.hstack((text_embedding,
                                                        text_entropy,
                                                        text_p_aggressive,
                                                        text_aggression,
                                                        text_is_decision_major,
                                                        text_overall_prc_mainstream))

                if pure_text_embeddings:
                    single_text_representation = text_embedding

                worker_embedding[index * single_text_representation_length: (index+1) * single_text_representation_length] = single_text_representation

            worker_embeddings[worker_id] = worker_embedding

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        if pure_text_embeddings:
            dir_name = 'new_pure_worker_embeddings'
        else:
            dir_name = 'new_worker_embeddings'
        with open(f'{dir_name}/new_worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)

    # random, equal class (0 and 1) distribution
    if scenario_number == '4':
        for i, worker_id in enumerate(df['worker_id'].unique()):

            # get positive and negative texts lists
            pos_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 0.0)]

            if len(pos_texts) >= (dev_texts // 2):
                if dev_texts == 1:
                    pos_worker_ids = pos_texts['rev_id'][:dev_texts].tolist()
                else:
                    pos_worker_ids = random.sample(pos_texts['rev_id'].tolist(), k=(dev_texts // 2))
                if dev_texts > 1 and dev_texts % 2 != 0:
                    pos_worker_ids = pos_worker_ids + random.sample(pos_texts['rev_id'].tolist(), k=1)
            else:
                additional_pos_texts = random.choices(pos_texts['rev_id'].tolist(), k=(dev_texts // 2) - len(pos_texts))
                pos_worker_ids = pos_texts['rev_id'].tolist() + additional_pos_texts

            neg_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 1.0)]

            if len(neg_texts) >= (dev_texts // 2):
                neg_worker_ids = random.sample(neg_texts['rev_id'].tolist(), k=(dev_texts // 2))
            else:
                additional_neg_texts = random.choices(neg_texts['rev_id'].tolist(), k=(dev_texts // 2) - len(neg_texts))
                neg_worker_ids = neg_texts['rev_id'].tolist() + additional_neg_texts

            worker_rev_ids = pos_worker_ids + neg_worker_ids

            single_text_representation_length = 305

            if pure_text_embeddings:
                single_text_representation_length = 300

            worker_embedding = np.empty((len(worker_rev_ids) * single_text_representation_length), dtype=np.float32)

            for index, rev_id in enumerate(worker_rev_ids):
                annotation_df = df[(df.rev_id == rev_id) & (df.worker_id == worker_id)]
                text_embedding = model.get_sentence_vector(annotation_df.comment.tolist()[0])
                text_entropy = annotation_df.entropy.tolist()[0]
                text_p_aggressive = annotation_df.p_aggressive.tolist()[0]
                text_aggression = annotation_df.aggression.tolist()[0]
                text_is_decision_major = annotation_df.is_decision_major.tolist()[0]
                text_overall_prc_mainstream = annotation_df.overall_prc_mainstream.tolist()[0]

                single_text_representation = np.hstack((text_embedding,
                                                        text_entropy,
                                                        text_p_aggressive,
                                                        text_aggression,
                                                        text_is_decision_major,
                                                        text_overall_prc_mainstream))

                worker_embedding[index * single_text_representation_length: (index+1) * single_text_representation_length] = single_text_representation

            worker_embeddings[worker_id] = worker_embedding

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        with open(f'new_worker_embeddings/new_worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)


def generate_worker_embeddings(scenario_number='1', dev_texts=20):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Reading data...')

    # load fasttext model
    model = fasttext.load_model('../cc.en.300.bin')

    # read dataframe with all the data
    df = pd.read_csv('../data/merged_folds_aggression.csv', sep=',', encoding='utf-8')

    # read JSON containing ids of dev split reviews for each worker
    with open('../data/docs_for_embeddings.json', 'r') as f:
        dev_rev_id = json.load(f)

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
          f'Generating personal embeddings for scenario: {scenario_number} | dev_texts: {dev_texts}...')

    worker_embeddings = defaultdict()

    # most controversial, random class (0 and 1) distribution
    if scenario_number == '1':
        for i, worker_id in enumerate(df['worker_id'].unique()):
            # get controversial-sorted sample of rev_ids for personal embeddings
            worker_dev_rev_ids = dev_rev_id[str(worker_id)]['controversial'][:dev_texts]

            # get positive and negative texts lists
            pos_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.rev_id.isin(worker_dev_rev_ids))
                           & (df.aggression == 0.0)].comment.tolist()
            neg_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.rev_id.isin(worker_dev_rev_ids))
                           & (df.aggression == 1.0)].comment.tolist()

            # get fasttext embeddings of texts
            pos_embedding = model.get_sentence_vector(' '.join(pos_texts))
            neg_embedding = model.get_sentence_vector(' '.join(neg_texts))

            worker_embeddings[worker_id] = {'positive': pos_embedding, 'negative': neg_embedding}

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        with open(f'worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)

    # most aggressive, random class (0 and 1) distribution
    if scenario_number == '2':
        for i, worker_id in enumerate(df['worker_id'].unique()):
            # get positive and negative texts lists
            texts = df[(df.worker_id == worker_id) & (df.split == 'dev')]
            texts = texts.sort_values(by='p_aggressive', ascending=False)
            texts = texts[:dev_texts]

            pos_texts = texts[texts['aggression'] == 0].comment.tolist()
            neg_texts = texts[texts['aggression'] == 1].comment.tolist()

            # get fasttext embeddings of texts
            pos_embedding = model.get_sentence_vector(' '.join(pos_texts))
            neg_embedding = model.get_sentence_vector(' '.join(neg_texts))

            worker_embeddings[worker_id] = {'positive': pos_embedding, 'negative': neg_embedding}

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        with open(f'worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)

    # most controversial, equal class (0 and 1) distribution
    if scenario_number == '3':
        for i, worker_id in enumerate(df['worker_id'].unique()):

            # get positive and negative texts lists
            pos_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 0.0)]
            pos_texts = pos_texts.sort_values(by='entropy', ascending=False)

            if len(pos_texts) >= (dev_texts // 2):
                pos_texts = pos_texts['comment'][:dev_texts//2]
            else:
                additional_pos_texts = random.choices(pos_texts['comment'].tolist(), k=(dev_texts // 2) - len(pos_texts))
                pos_texts = pos_texts['comment'].tolist() + additional_pos_texts

            neg_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 1.0)]
            neg_texts = neg_texts.sort_values(by='entropy', ascending=False)

            if len(neg_texts) >= (dev_texts // 2):
                neg_texts = neg_texts['comment'][:dev_texts//2]
            else:
                additional_pos_texts = random.choices(neg_texts['comment'].tolist(), k=(dev_texts // 2) - len(neg_texts))
                neg_texts = neg_texts['comment'].tolist() + additional_pos_texts

            # get fasttext embeddings of texts
            pos_embedding = model.get_sentence_vector(' '.join(pos_texts))
            neg_embedding = model.get_sentence_vector(' '.join(neg_texts))

            worker_embeddings[worker_id] = {'positive': pos_embedding, 'negative': neg_embedding}

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        with open(f'worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)

    # random, equal class (0 and 1) distribution
    if scenario_number == '4':
        for i, worker_id in enumerate(df['worker_id'].unique()):

            # get positive and negative texts lists
            pos_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 0.0)]

            if len(pos_texts) >= (dev_texts // 2):
                pos_texts = random.sample(pos_texts['comment'].tolist(), k=(dev_texts // 2))
            else:
                additional_pos_texts = random.choices(pos_texts['comment'].tolist(), k=(dev_texts // 2) - len(pos_texts))
                pos_texts = pos_texts['comment'].tolist() + additional_pos_texts

            neg_texts = df[(df.worker_id == worker_id) & (df.split == 'dev') & (df.aggression == 1.0)]

            if len(neg_texts) >= (dev_texts // 2):
                neg_texts = random.sample(neg_texts['comment'].tolist(), k=(dev_texts // 2))
            else:
                additional_pos_texts = random.choices(neg_texts['comment'].tolist(), k=(dev_texts // 2) - len(neg_texts))
                neg_texts = neg_texts['comment'].tolist() + additional_pos_texts

            # get fasttext embeddings of texts
            pos_embedding = model.get_sentence_vector(' '.join(pos_texts))
            neg_embedding = model.get_sentence_vector(' '.join(neg_texts))

            worker_embeddings[worker_id] = {'positive': pos_embedding, 'negative': neg_embedding}

            if i % 100 == 0:
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                      f'Processed {i} / {len(df["worker_id"].unique())} workers')

        # save personal embeddings to file
        with open(f'worker_embeddings/scenario_{scenario_number}/worker_embeddings_{scenario_number}_{dev_texts}.pkl', 'wb') as f:
            pkl.dump(worker_embeddings, f, protocol=4)


def prepare_data(model, df, scenario_number="1", dev_texts=20, test_fold_num=10, new_embeddings=False, pure_embeddings=False):

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
          f'Generating training data for',
          f'scenario: {scenario_number} | dev_texts: {dev_texts} | test_fold_num: {test_fold_num}...')

    # read personal embeddings
    embeddings_path = f'worker_embeddings/scenario_{scenario_number}/worker_embeddings_{scenario_number}_{dev_texts}.pkl'

    if new_embeddings:
        embeddings_path = f'new_worker_embeddings/new_worker_embeddings_{scenario_number}_{dev_texts}.pkl'

    if pure_embeddings:
        embeddings_path = f'new_pure_worker_embeddings/new_worker_embeddings_{scenario_number}_{dev_texts}.pkl'

    with open(embeddings_path, 'rb') as f:
        worker_embeddings = pkl.load(f)

    folds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    test_fold = [folds[test_fold_num-1]]

    if new_embeddings:
        embedding_length = 300 + dev_texts * 305
    else:
        if pure_embeddings:
            embedding_length = 300 + dev_texts * 300
        else:
            embedding_length = 300 + 600

    X = {'train': np.empty((len(df[(df.fold != test_fold[0]) & (df.split == 'train')]), embedding_length)),
         'test': np.empty((len(df[(df.fold == test_fold[0]) & (df.split == 'test')]), embedding_length))}
    # y = {'train': np.empty(len(df[(df.fold != test_fold[0]) & (df.split == 'train')]), dtype=np.int32),
    #      'test': np.empty(len(df[(df.fold == test_fold[0]) & (df.split == 'test')]), dtype=np.int32)}

    # create dataframe containing train data
    df_train = df[(df.fold != test_fold[0]) & (df.split == 'train')]

    # create dataframe containing test data
    # df_test = df[(df.fold == test_fold[0]) & (df.split == 'test')]

    # initialize train X and train y lists index
    sample_index = 0
    for index, row in df_train.iterrows():

        # get input embeddings
        text_embedding = model.get_sentence_vector(row['comment'])

        if pure_embeddings:
            personal_embedding = worker_embeddings[row['worker_id']]
            print(f'personal_embedding shape: {personal_embedding.shape}',
                  f'x train shape: {X["train"][sample_index].shape}',
                  sep='\n')
            X['train'][sample_index] = np.hstack((text_embedding, personal_embedding))

        if new_embeddings:
            personal_embedding = worker_embeddings[row['worker_id']]
            X['train'][sample_index] = np.hstack((text_embedding, personal_embedding))

        if not new_embeddings and not pure_embeddings:
            pos_embedding = worker_embeddings[row['worker_id']]['positive']
            neg_embedding = worker_embeddings[row['worker_id']]['negative']
            X['train'][sample_index] = np.hstack((text_embedding, pos_embedding, neg_embedding))

        sample_index += 1

        if row['aggression'] not in [0, 1]:
            print(f'Invalid label: {row["aggression"]}')

    # initialize test X and test y lists index
    # sample_index = 0
    # for index, row in df_test.iterrows():
    #
    #     # get input embeddings
    #     text_embedding = model.get_sentence_vector(row['comment'])
    #     pos_embedding = worker_embeddings[row['worker_id']]['positive']
    #     neg_embedding = worker_embeddings[row['worker_id']]['negative']
    #
    #     X['test'][sample_index] = np.hstack((text_embedding, pos_embedding, neg_embedding))
    #
    #     sample_index += 1
    #
    #     if row['aggression'] not in [0, 1]:
    #         print(f'Invalid label: {row["aggression"]}')

    # train model and save it
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Training Logistic Regression model...')
    X_train = X['train']

    # X_test = X['test']

    y_train = df_train['aggression']

    # y_test = df_test['aggression']

    lr = LogisticRegression(max_iter=2000, verbose=1)

    start_time = time.time()

    lr.fit(X_train, y_train)

    end_time = time.time()
    print(f'{"*" * 10}TIME OF LEARNING ON SCENARIO {scenario_number} | DEV TEXTS {dev_texts} : {end_time - start_time}')

    model_path = f'models/scenario_{scenario_number}/lr_model_{scenario_number}_{dev_texts}_{test_fold_num}.pkl'

    if new_embeddings:
        model_path = f'new_embeddings_models/scenario_{scenario_number}/lr_model_{scenario_number}_{dev_texts}_{test_fold_num}.pkl'

    if pure_embeddings:
        model_path = f'new_pure_embeddings_models/scenario_{scenario_number}/lr_model_{scenario_number}_{dev_texts}_{test_fold_num}.pkl'

    with open(model_path, 'wb') as f:
        pkl.dump(lr, f, protocol=4)

    # y_pred = lr.predict(X_test)

    # report = classification_report(y_test, y_pred, output_dict=True, digits=4)

    # with open(f'results_{scenario_number}_{dev_texts}_{test_fold_num}.pkl', 'wb') as f:
    #     pkl.dump(report, f, protocol=4)
    # print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Saving samples and labels...')

    # save X and y
    # with open(f'X_train_{scenario_number}_{dev_texts}_{test_fold_num}.pkl', 'wb') as f:
    #     pkl.dump(X['train'], f, protocol=4)
    # df_train.to_csv(f'y_train_{scenario_number}_{dev_texts}_{test_fold_num}.csv', columns=['aggression'], index=False)
    # with open(f'X_test_{scenario_number}_{dev_texts}_{test_fold_num}.pkl', 'wb') as f:
    #     pkl.dump(X['test'], f, protocol=4)
    # df_test.to_csv(f'y_test_{scenario_number}_{dev_texts}_{test_fold_num}.csv', columns=['aggression'], index=False)


def evaluate():
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : ***EVALUATE***')
    results = dict()

    df_train = pd.read_csv('labels_train.csv', sep=',', encoding='utf-8')
    df_test = pd.read_csv('labels_test.csv', sep=',', encoding='utf-8')

    # for i in range(10):
    # print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : ****Iteration: {i}')
    with open('embeddings_train.pkl', 'rb') as f:
        X_train = pkl.load(f)

    with open('embeddings_test.pkl', 'rb') as f:
        X_test = pkl.load(f)

    y_train = df_train['aggression']

    y_test = df_test['aggression']

    # N = X_train.shape[0]
    # Nfrac = int(N * 0.9)
    #
    # X_train, y_train = X_train[:Nfrac, :], y_train[:Nfrac]

    lr = LogisticRegression(max_iter=2000, verbose=1)
    lr.fit(X_train, y_train)
    with open(f'lr_model.pkl', 'wb') as f:
        pkl.dump(lr, f, protocol=4)

    y_pred = lr.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    print(classification_report(y_test, y_pred, digits=4))
    results = report
    print()

    with open('results.dat', 'wb') as f:
        pkl.dump(results, f, protocol=4)


def stats_calc():
    alpha = 0.05
    samples = dict()
    with open('results.dat', 'rb') as f:
        results = pkl.load(f)
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
                # p = round(p, 6)
                if p > 0.05:
                    result05 = 'Gaussian'
                else:
                    result05 = 'Fail'
                if p > 0.01:
                    result01 = 'Gaussian'
                else:
                    result01 = 'Fail'
                # print(f'Set: {s},  Measure: {measure}, Shapiro-Wilk: p-val:{p}, for alpha=0.05: {result05}, for alpha=0.01: {result01}')
                row_format = "{:>25}" * 5
                print(row_format.format(s, measure, p, result05, result01))

    # for sa, sb in [('s1', 's2'), ('s1', 's3'), ('s1', 's4'), ('s2', 's3'), ('s2', 's4'), ('s3', 's4')]:
    for sa, sb in [('s1', 's2'), ('s1', 's3'), ('s1', 's4'), ('s2', 's3'), ('s3', 's4')]:
        print(f'Pair: ({sa},{sb})')
        for measure in ['accuracy', 'p-macro', 'r-macro', 'f1-macro', 'p-a', 'r-a', 'f1-a']:
            stat, p = stats.ttest_rel(samples[sa][measure], samples[sb][measure])
            # p = round(p, 6)
            if p > 0.05:
                result05 = 'Fail'
            else:
                result05 = 'Significant'
            if p > 0.01:
                result01 = 'Fail'
            else:
                result01 = 'Significant'
            # print(f'Measure: {measure}, Paired sample t-test:  p-val: {p}, for alpha=0.05: {result05}, for alpha=0.01: {result01}')
            row_format = "{:>25}" * 6
            print(row_format.format(sa, sb, measure, p, result05, result01))

    # for s in ['s1', 's2', 's3', 's4']:
    #    for measure in ['accuracy', 'p-macro', 'r-macro', 'f1-macro', 'p-a', 'r-a', 'f1-a']:
    #        row_format = "{:>25}" * 4


if __name__ == '__main__':
    scenario = sys.argv[1]

    # generate worker embeddings
    # for texts_num in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    # for texts_num in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    # for texts_num in [3, 5, 7, 9, 11, 13, 15, 17, 19]:
    #     generate_new_worker_embeddings(str(scenario), texts_num, pure_text_embeddings=False)
    #     generate_worker_embeddings(str(scenario), texts_num)

    # load fasttext model
    model_ft = fasttext.load_model('../cc.en.300.bin')

    # read dataframe with all the data
    df_aggression = pd.read_csv('../data/merged_folds_aggression.csv', sep=',', encoding='utf-8')

    # for texts_num in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    # for texts_num in [3, 5, 30, 32, 34, 36, 38, 40]:
    # for texts_num in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
    # for texts_num in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    for texts_num in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    #     # folds_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        folds_num = [1]
        for test_fold in folds_num:
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : getting results for model {texts_num} | {test_fold}...')
            prepare_data(model_ft, df_aggression, str(scenario), texts_num, test_fold, new_embeddings=True, pure_embeddings=False)

    # prepare_data(dev_texts=20)
    # evaluate()
    # stats_calc()
