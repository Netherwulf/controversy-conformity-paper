#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import pickle as pkl

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


# In[79]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3]:
        for dev_texts in [3, 4, 5]:

            for fold in [1]:
                for interval in ['0_20', '20_40', '40_60', '60_80', '80_100']:
                    with open(f'../data/partly_controversial_results/{interval}/scenario_{scenario_number}/controversial_{interval}_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                        data = pkl.load(f)

                    result_df = result_df.append({'scenario': scenario_number,
                                                  'dev_texts': dev_texts,
                                                  'fold': fold,
                                                  'interval': interval,
                                                  '0_precision': data['0']['precision'],
                                                  '0_recall': data['0']['recall'],
                                                  '0_f1-score': data['0']['f1-score'],
                                                  '0_support': data['0']['support'],
                                                  '1_precision': data['1']['precision'],
                                                  '1_recall': data['1']['recall'],
                                                  '1_f1-score': data['1']['f1-score'],
                                                  '1_support': data['1']['support'],
                                                  'accuracy': data['accuracy'],
                                                  'macro_avg_precision': data['macro avg']['precision'],
                                                  'macro_avg_recall': data['macro avg']['recall'],
                                                  'macro_avg_f1-score': data['macro avg']['f1-score'],
                                                  'macro_avg_support': data['macro avg']['support'],
                                                  'weighted_avg_precision': data['weighted avg']['precision'],
                                                  'weighted_avg_recall': data['weighted avg']['recall'],
                                                  'weighted_avg_f1-score': data['weighted avg']['f1-score'],
                                                  'weighted_avg_support': data['weighted avg']['support']}, 
                                                 ignore_index=True)
            
    return result_df


# In[80]:


df = get_scenario_results()
df


# <h3>Plots for Scenario (1, 2, 3) with dev texts = [3, 4, 5]</h3>

# In[81]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty((5, 3), dtype=np.float32)
avg_f1_scores_s_2 = np.empty((5, 3), dtype=np.float32)
avg_f1_scores_s_3 = np.empty((5, 3), dtype=np.float32)


for i, interval in enumerate(['0_20', '20_40', '40_60', '60_80', '80_100']):
    for dev_texts in [3, 4, 5]:
        avg_f1_scores_s_1[i][dev_texts - 3] = df[(df.scenario == 1) & (df.dev_texts == dev_texts) & (df.interval == interval)]['macro_avg_f1-score'].mean()
        avg_f1_scores_s_2[i][dev_texts - 3] = df[(df.scenario == 2) & (df.dev_texts == dev_texts) & (df.interval == interval)]['macro_avg_f1-score'].mean()
        avg_f1_scores_s_3[i][dev_texts - 3] = df[(df.scenario == 3) & (df.dev_texts == dev_texts) & (df.interval == interval)]['macro_avg_f1-score'].mean()

# Set position of bar on X axis
r1 = np.arange(1, 18)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3]

colors = ['r', 'g', 'b', 'm', 'c']

for i, f1_scores in enumerate(avg_f1_scores):
    for interval in range(0, 5):
            plt.plot(r1[i+i*3: i+3+i*3],
                     f1_scores[interval],
                     f'{colors[interval]}o-',
                     label=f'{interval*20}% - {interval*20+20}% most controversial')

plt.title('F1-score for 3/4/5 dev texts for each scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(2, 14, 4)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution'])

legend_elements = [Line2D([0], [0], color='c', lw=4, label='80% - 100% most controversial'),
                   Line2D([0], [0], color='m', lw=4, label='60% - 80% most controversial'),
                   Line2D([0], [0], color='b', lw=4, label='40% - 60% most controversial'),
                   Line2D([0], [0], color='g', lw=4, label='20% - 40% most controversial'),
                   Line2D([0], [0], color='r', lw=4, label='0% - 20% most controversial')]
plt.legend(handles=legend_elements)

# plt.ylim(0.21, 0.32)
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[85]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty((5, 3), dtype=np.float32)
avg_f1_scores_s_2 = np.empty((5, 3), dtype=np.float32)
avg_f1_scores_s_3 = np.empty((5, 3), dtype=np.float32)


for i, interval in enumerate(['0_20', '20_40', '40_60', '60_80', '80_100']):
    for dev_texts in [3, 4, 5]:
        avg_f1_scores_s_1[i][dev_texts - 3] = df[(df.scenario == 1) & (df.dev_texts == dev_texts) & (df.interval == interval)]['weighted_avg_f1-score'].mean()
        avg_f1_scores_s_2[i][dev_texts - 3] = df[(df.scenario == 2) & (df.dev_texts == dev_texts) & (df.interval == interval)]['weighted_avg_f1-score'].mean()
        avg_f1_scores_s_3[i][dev_texts - 3] = df[(df.scenario == 3) & (df.dev_texts == dev_texts) & (df.interval == interval)]['weighted_avg_f1-score'].mean()

# Set position of bar on X axis
r1 = np.arange(1, 18)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3]

colors = ['r', 'g', 'b', 'm', 'c']

for i, f1_scores in enumerate(avg_f1_scores):
    for interval in range(0, 5):
            plt.plot(r1[i+i*3: i+3+i*3],
                     f1_scores[interval],
                     f'{colors[interval]}o-',
                     label=f'{interval*20}% - {interval*20+20}% most controversial')

plt.title('Weighted F1-score for 3/4/5 dev texts for each scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(2, 14, 4)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution'])

legend_elements = [Line2D([0], [0], color='c', lw=4, label='80% - 100% most controversial'),
                   Line2D([0], [0], color='m', lw=4, label='60% - 80% most controversial'),
                   Line2D([0], [0], color='b', lw=4, label='40% - 60% most controversial'),
                   Line2D([0], [0], color='g', lw=4, label='20% - 40% most controversial'),
                   Line2D([0], [0], color='r', lw=4, label='0% - 20% most controversial')]
plt.legend(handles=legend_elements)

# plt.ylim(0.21, 0.32)
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[86]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty((5, 3), dtype=np.float32)
avg_f1_scores_s_2 = np.empty((5, 3), dtype=np.float32)
avg_f1_scores_s_3 = np.empty((5, 3), dtype=np.float32)


for i, interval in enumerate(['0_20', '20_40', '40_60', '60_80', '80_100']):
    for dev_texts in [3, 4, 5]:
        avg_f1_scores_s_1[i][dev_texts - 3] = df[(df.scenario == 1) & (df.dev_texts == dev_texts) & (df.interval == interval)]['accuracy'].mean()
        avg_f1_scores_s_2[i][dev_texts - 3] = df[(df.scenario == 2) & (df.dev_texts == dev_texts) & (df.interval == interval)]['accuracy'].mean()
        avg_f1_scores_s_3[i][dev_texts - 3] = df[(df.scenario == 3) & (df.dev_texts == dev_texts) & (df.interval == interval)]['accuracy'].mean()

# Set position of bar on X axis
r1 = np.arange(1, 18)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3]

colors = ['r', 'g', 'b', 'm', 'c']

for i, f1_scores in enumerate(avg_f1_scores):
    for interval in range(0, 5):
            plt.plot(r1[i+i*3: i+3+i*3],
                     f1_scores[interval],
                     f'{colors[interval]}o-',
                     label=f'{interval*20}% - {interval*20+20}% most controversial')

plt.title('Accuracy for 3/4/5 dev texts for each scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(2, 14, 4)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution'])

legend_elements = [Line2D([0], [0], color='c', lw=4, label='80% - 100% most controversial'),
                   Line2D([0], [0], color='m', lw=4, label='60% - 80% most controversial'),
                   Line2D([0], [0], color='b', lw=4, label='40% - 60% most controversial'),
                   Line2D([0], [0], color='g', lw=4, label='20% - 40% most controversial'),
                   Line2D([0], [0], color='r', lw=4, label='0% - 20% most controversial')]
plt.legend(handles=legend_elements)

# plt.ylim(0.21, 0.32)
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[87]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [2, 3]:
        for dev_texts in list(range(0, 21, 2)) + list(range(30, 41, 2)):

            for fold in [1]:
                for interval in ['0_20', '20_40', '40_60', '60_80', '80_100']:
                    with open(f'../data/partly_controversial_results/{interval}/scenario_{scenario_number}/controversial_{interval}_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                        data = pkl.load(f)

                    result_df = result_df.append({'scenario': scenario_number,
                                                  'dev_texts': dev_texts,
                                                  'fold': fold,
                                                  'interval': interval,
                                                  '0_precision': data['0']['precision'],
                                                  '0_recall': data['0']['recall'],
                                                  '0_f1-score': data['0']['f1-score'],
                                                  '0_support': data['0']['support'],
                                                  '1_precision': data['1']['precision'],
                                                  '1_recall': data['1']['recall'],
                                                  '1_f1-score': data['1']['f1-score'],
                                                  '1_support': data['1']['support'],
                                                  'accuracy': data['accuracy'],
                                                  'macro_avg_precision': data['macro avg']['precision'],
                                                  'macro_avg_recall': data['macro avg']['recall'],
                                                  'macro_avg_f1-score': data['macro avg']['f1-score'],
                                                  'macro_avg_support': data['macro avg']['support'],
                                                  'weighted_avg_precision': data['weighted avg']['precision'],
                                                  'weighted_avg_recall': data['weighted avg']['recall'],
                                                  'weighted_avg_f1-score': data['weighted avg']['f1-score'],
                                                  'weighted_avg_support': data['weighted avg']['support']}, 
                                                 ignore_index=True)
            
    return result_df


# In[88]:


df = get_scenario_results()
df


# In[78]:


# macro f1-score for each scenario
avg_f1_scores_s_2 = np.empty((5, 17), dtype=np.float32)
avg_f1_scores_s_3 = np.empty((5, 17), dtype=np.float32)


for i, interval in enumerate(['0_20', '20_40', '40_60', '60_80', '80_100']):
    for j, dev_texts in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 32, 34, 36, 38, 40]):
        avg_f1_scores_s_2[i][j] = df[(df.scenario == 2) & (df.dev_texts == dev_texts) & (df.interval == interval)]['macro_avg_f1-score'].mean()
        avg_f1_scores_s_3[i][j] = df[(df.scenario == 3) & (df.dev_texts == dev_texts) & (df.interval == interval)]['macro_avg_f1-score'].mean()

# Set position of bar on X axis
r1 = np.arange(1, 35)

avg_f1_scores = [avg_f1_scores_s_2, avg_f1_scores_s_3]

colors = ['r', 'g', 'b', 'm', 'c']

for i, f1_scores in enumerate(avg_f1_scores):
    for interval in range(0, 5):
            plt.plot(r1[i+i*16: i+17+i*16],
                     f1_scores[interval],
                     f'{colors[interval]}o-',
                     label=f'{interval*20}% - {interval*20+20}% most controversial')

plt.title('F1-score for 0/2/4/6/8/10/12/14/16/18/20 and 30/32/34/36/38/40 dev texts for 2nd and 3rd scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(10, 28, 17)), ['2 - most aggressive,\n random class distribution',
                                    '3 - most controversial,\n equal class distribution'])

legend_elements = [Line2D([0], [0], color='c', lw=4, label='80% - 100% most controversial'),
                   Line2D([0], [0], color='m', lw=4, label='60% - 80% most controversial'),
                   Line2D([0], [0], color='b', lw=4, label='40% - 60% most controversial'),
                   Line2D([0], [0], color='g', lw=4, label='20% - 40% most controversial'),
                   Line2D([0], [0], color='r', lw=4, label='0% - 20% most controversial')]
plt.legend(handles=legend_elements)

# plt.ylim(0.21, 0.32)
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[90]:


# macro f1-score for each scenario
avg_f1_scores_s_2 = np.empty((5, 17), dtype=np.float32)
avg_f1_scores_s_3 = np.empty((5, 17), dtype=np.float32)


for i, interval in enumerate(['0_20', '20_40', '40_60', '60_80', '80_100']):
    for j, dev_texts in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 32, 34, 36, 38, 40]):
        avg_f1_scores_s_2[i][j] = df[(df.scenario == 2) & (df.dev_texts == dev_texts) & (df.interval == interval)]['weighted_avg_f1-score'].mean()
        avg_f1_scores_s_3[i][j] = df[(df.scenario == 3) & (df.dev_texts == dev_texts) & (df.interval == interval)]['weighted_avg_f1-score'].mean()

# Set position of bar on X axis
r1 = np.arange(1, 35)

avg_f1_scores = [avg_f1_scores_s_2, avg_f1_scores_s_3]

colors = ['r', 'g', 'b', 'm', 'c']

for i, f1_scores in enumerate(avg_f1_scores):
    for interval in range(0, 5):
            plt.plot(r1[i+i*16: i+17+i*16],
                     f1_scores[interval],
                     f'{colors[interval]}o-',
                     label=f'{interval*20}% - {interval*20+20}% most controversial')

plt.title('Weighted F1-score for 0/2/4/6/8/10/12/14/16/18/20 and 30/32/34/36/38/40 dev texts for 2nd and 3rd scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(10, 28, 17)), ['2 - most aggressive,\n random class distribution',
                                    '3 - most controversial,\n equal class distribution'])

legend_elements = [Line2D([0], [0], color='c', lw=4, label='80% - 100% most controversial'),
                   Line2D([0], [0], color='m', lw=4, label='60% - 80% most controversial'),
                   Line2D([0], [0], color='b', lw=4, label='40% - 60% most controversial'),
                   Line2D([0], [0], color='g', lw=4, label='20% - 40% most controversial'),
                   Line2D([0], [0], color='r', lw=4, label='0% - 20% most controversial')]
plt.legend(handles=legend_elements)

# plt.ylim(0.21, 0.32)
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[91]:


# macro f1-score for each scenario
avg_f1_scores_s_2 = np.empty((5, 17), dtype=np.float32)
avg_f1_scores_s_3 = np.empty((5, 17), dtype=np.float32)


for i, interval in enumerate(['0_20', '20_40', '40_60', '60_80', '80_100']):
    for j, dev_texts in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 32, 34, 36, 38, 40]):
        avg_f1_scores_s_2[i][j] = df[(df.scenario == 2) & (df.dev_texts == dev_texts) & (df.interval == interval)]['accuracy'].mean()
        avg_f1_scores_s_3[i][j] = df[(df.scenario == 3) & (df.dev_texts == dev_texts) & (df.interval == interval)]['accuracy'].mean()

# Set position of bar on X axis
r1 = np.arange(1, 35)

avg_f1_scores = [avg_f1_scores_s_2, avg_f1_scores_s_3]

colors = ['r', 'g', 'b', 'm', 'c']

for i, f1_scores in enumerate(avg_f1_scores):
    for interval in range(0, 5):
            plt.plot(r1[i+i*16: i+17+i*16],
                     f1_scores[interval],
                     f'{colors[interval]}o-',
                     label=f'{interval*20}% - {interval*20+20}% most controversial')

plt.title('Accuracy for 0/2/4/6/8/10/12/14/16/18/20 and 30/32/34/36/38/40 dev texts for 2nd and 3rd scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(10, 28, 17)), ['2 - most aggressive,\n random class distribution',
                                    '3 - most controversial,\n equal class distribution'])

legend_elements = [Line2D([0], [0], color='c', lw=4, label='80% - 100% most controversial'),
                   Line2D([0], [0], color='m', lw=4, label='60% - 80% most controversial'),
                   Line2D([0], [0], color='b', lw=4, label='40% - 60% most controversial'),
                   Line2D([0], [0], color='g', lw=4, label='20% - 40% most controversial'),
                   Line2D([0], [0], color='r', lw=4, label='0% - 20% most controversial')]
plt.legend(handles=legend_elements)

# plt.ylim(0.21, 0.32)
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()

