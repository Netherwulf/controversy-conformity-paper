#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import pickle as pkl
# import matplotlib
# matplotlib.use('PS')
# %matplotlib inline
from matplotlib import pyplot as plt


# In[2]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3, 4]:
        for dev_texts in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

                with open(f'../data/results/scenario_{scenario_number}/results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                    data = pkl.load(f)

                result_df = result_df.append({'scenario': scenario_number,
                                              'dev_texts': dev_texts,
                                              'fold': fold,
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


# In[3]:


df = get_scenario_results()
df


# <h3>Scenario 1 plots (most controversial, random class (0 and 1) distribution)</h3>

# In[73]:


# macro f1-score plot for scenario 1
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Macro F1-score (1 - most controversial, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.67, 0.72)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()



# weighted f1-score plot for scenario 1
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Weighted F1-score (1 - most controversial, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.82, 0.845)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()



# accuracy plot for scenario 1
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Accuracy (1 - most controversial, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.84, 0.865)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# macro f1-score plot for scenario 2
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Macro F1-score (2 - most aggressive, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.67, 0.71)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# In[84]:


# weighted f1-score plot for scenario 2
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Weighted F1-score (2 - most aggressive, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.825, 0.84)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# In[86]:


# accuracy plot for scenario 2
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Accuracy (2 - most aggressive, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.845, 0.860)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# <h3>Scenario 3 plots (most controversial, equal class (0 and 1) distribution)</h3>

# In[87]:


# macro f1-score plot for scenario 3
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Macro F1-score (3 - most controversial, equal class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.67, 0.71)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# In[91]:


# weighted f1-score plot for scenario 3
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Weighted F1-score (3 - most controversial, equal class (0 and 1) distribution))', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.825, 0.845)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# In[95]:


# accuracy plot for scenario 3
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Accuracy (3 - most controversial, equal class (0 and 1) distribution))', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.85, 0.865)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()
#

# macro f1-score plot for scenario 4
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Macro F1-score (4 - random, equal class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.67, 0.71)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# In[100]:


# weighted f1-score plot for scenario 4
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Weighted F1-score (4 - random, equal class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.825, 0.84)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()


# In[102]:


# accuracy plot for scenario 4
dev_texts_list = np.empty(11, dtype=np.int64)
avg_f1_scores = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    dev_texts_list[dev_texts // 2] = dev_texts
    avg_f1_scores[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

# Function to plot
plt.bar(dev_texts_list, avg_f1_scores)

plt.title('Accuracy (4 - random, equal class (0 and 1) distribution)', fontsize=15)
plt.xlabel("Number of texts in dev", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(0, 21, 2)), dev_texts_list)
plt.ylim(0.85, 0.86)
fig = plt.gcf()
fig.set_size_inches(11,8)

# function to show the plot
plt.show()

# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Macro F1-score for each scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.68, 0.715)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()



# weighted f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Weighted F1-score for each scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.82, 0.85)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# In[135]:


# accuracy for each scenario
accuracy_scores_s_1 = np.empty(11, dtype=np.float32)
accuracy_scores_s_2 = np.empty(11, dtype=np.float32)
accuracy_scores_s_3 = np.empty(11, dtype=np.float32)
accuracy_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

accuracy_scores = [accuracy_scores_s_1, accuracy_scores_s_2, accuracy_scores_s_3, accuracy_scores_s_4]

plt.bar(r1, [score[0] for score in accuracy_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in accuracy_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in accuracy_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in accuracy_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in accuracy_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in accuracy_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in accuracy_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in accuracy_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in accuracy_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in accuracy_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in accuracy_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Accuracy for each scenario', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.85, 0.865)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()

# box plot for 1st scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Macro F1-score (1 - most controversial, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 2nd scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Macro F1-score (2 - most aggressive, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 3rd scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Macro F1-score (3 - most controversial, equal class (0 and 1) distribution))', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 4th scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Macro F1-score (4 - random, equal class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 1st scenario (weighted f1-score)
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Weighted F1-score (1 - most controversial, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 2nd scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Weighted F1-score (2 - most aggressive, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 3rd scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Weighted F1-score (3 - most controversial, equal class (0 and 1) distribution))', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 4th scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Weighted F1-score (4 - random, equal class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 1st scenario (accuracy)
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Accuracy (1 - most controversial, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 2nd scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Accuracy (2 - most aggressive, random class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 3rd scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Accuracy (3 - most controversial, equal class (0 and 1) distribution))', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()

# box plot for 4th scenario
accuracy_scores_s_1 = np.empty((11, 10), dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].tolist()

ticks = np.empty(11, dtype=np.int64)

for box_id in range(0, 11):
    plt.boxplot(accuracy_scores_s_1[box_id], positions=[box_id], widths=0.5)
    ticks[box_id] = box_id

plt.title('Accuracy (4 - random, equal class (0 and 1) distribution)', fontsize=15)
plt.xlabel("dev texts", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xlim(-1,ticks[-1]+1) # need to shift the right end of the x limit by 1
plt.xticks(ticks, [str(x*2) for x in ticks])
fig = plt.gcf()
fig.set_size_inches(14,10)

fig.tight_layout()

plt.show()


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3, 4]:
        for dev_texts in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

                with open(f'../data/controversial_results/scenario_{scenario_number}/controversial_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                    data = pkl.load(f)

                result_df = result_df.append({'scenario': scenario_number,
                                              'dev_texts': dev_texts,
                                              'fold': fold,
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


# In[16]:


import os
arr = os.listdir('../JIPM_LR')
print(arr)


# In[18]:


df = get_scenario_results()
df


# In[23]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Macro F1-score for each scenario (entropy >= 0.1)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.59, 0.64)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[28]:


# weighted f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Weighted F1-score for each scenario (entropy >= 0.1)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.67, 0.72)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# In[32]:


# accuracy for each scenario
accuracy_scores_s_1 = np.empty(11, dtype=np.float32)
accuracy_scores_s_2 = np.empty(11, dtype=np.float32)
accuracy_scores_s_3 = np.empty(11, dtype=np.float32)
accuracy_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

accuracy_scores = [accuracy_scores_s_1, accuracy_scores_s_2, accuracy_scores_s_3, accuracy_scores_s_4]

plt.bar(r1, [score[0] for score in accuracy_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in accuracy_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in accuracy_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in accuracy_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in accuracy_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in accuracy_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in accuracy_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in accuracy_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in accuracy_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in accuracy_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in accuracy_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Accuracy for each scenario (entropy >= 0.1)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.73, 0.755)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# <h3>Results on 0%-20% most controversial texts</h3>

# In[37]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3, 4]:
        for dev_texts in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

                with open(f'../data/partly_controversial_results/0_20/scenario_{scenario_number}/controversial_0_20_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                    data = pkl.load(f)

                result_df = result_df.append({'scenario': scenario_number,
                                              'dev_texts': dev_texts,
                                              'fold': fold,
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


# In[39]:


df = get_scenario_results()
df


# In[43]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Macro F1-score for each scenario (0% - 20% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.42, 0.51)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[50]:


# weighted f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Weighted F1-score for each scenario (0% - 20% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.44, 0.52)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# In[53]:


# accuracy for each scenario
accuracy_scores_s_1 = np.empty(11, dtype=np.float32)
accuracy_scores_s_2 = np.empty(11, dtype=np.float32)
accuracy_scores_s_3 = np.empty(11, dtype=np.float32)
accuracy_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

accuracy_scores = [accuracy_scores_s_1, accuracy_scores_s_2, accuracy_scores_s_3, accuracy_scores_s_4]

plt.bar(r1, [score[0] for score in accuracy_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in accuracy_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in accuracy_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in accuracy_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in accuracy_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in accuracy_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in accuracy_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in accuracy_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in accuracy_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in accuracy_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in accuracy_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Accuracy for each scenario (0% - 20% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.52, 0.57)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# <h3>Results on 20%-40% most controversial texts</h3>

# In[54]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3, 4]:
        for dev_texts in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

                with open(f'../data/partly_controversial_results/20_40/scenario_{scenario_number}/controversial_20_40_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                    data = pkl.load(f)

                result_df = result_df.append({'scenario': scenario_number,
                                              'dev_texts': dev_texts,
                                              'fold': fold,
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


# In[55]:


df = get_scenario_results()
df


# In[62]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Macro F1-score for each scenario (20% - 40% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.52, 0.58)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[69]:


# weighted f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Weighted F1-score for each scenario (20% - 40% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.575, 0.625)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# In[72]:


# accuracy for each scenario
accuracy_scores_s_1 = np.empty(11, dtype=np.float32)
accuracy_scores_s_2 = np.empty(11, dtype=np.float32)
accuracy_scores_s_3 = np.empty(11, dtype=np.float32)
accuracy_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

accuracy_scores = [accuracy_scores_s_1, accuracy_scores_s_2, accuracy_scores_s_3, accuracy_scores_s_4]

plt.bar(r1, [score[0] for score in accuracy_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in accuracy_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in accuracy_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in accuracy_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in accuracy_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in accuracy_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in accuracy_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in accuracy_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in accuracy_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in accuracy_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in accuracy_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Accuracy for each scenario (20% - 40% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.63, 0.67)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# <h3>Results on 40%-60% most controversial texts</h3>

# In[75]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3, 4]:
        for dev_texts in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

                with open(f'../data/partly_controversial_results/40_60/scenario_{scenario_number}/controversial_40_60_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                    data = pkl.load(f)

                result_df = result_df.append({'scenario': scenario_number,
                                              'dev_texts': dev_texts,
                                              'fold': fold,
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


# In[76]:


df = get_scenario_results()
df


# In[80]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Macro F1-score for each scenario (40% - 60% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.61, 0.655)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[84]:


# weighted f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Weighted F1-score for each scenario (40% - 60% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.71, 0.74)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# In[90]:


# accuracy for each scenario
accuracy_scores_s_1 = np.empty(11, dtype=np.float32)
accuracy_scores_s_2 = np.empty(11, dtype=np.float32)
accuracy_scores_s_3 = np.empty(11, dtype=np.float32)
accuracy_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

accuracy_scores = [accuracy_scores_s_1, accuracy_scores_s_2, accuracy_scores_s_3, accuracy_scores_s_4]

plt.bar(r1, [score[0] for score in accuracy_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in accuracy_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in accuracy_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in accuracy_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in accuracy_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in accuracy_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in accuracy_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in accuracy_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in accuracy_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in accuracy_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in accuracy_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Accuracy for each scenario (40% - 60% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.75, 0.775)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# <h3>Results on 60%-80% most controversial texts</h3>

# In[92]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3, 4]:
        for dev_texts in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

                with open(f'../data/partly_controversial_results/60_80/scenario_{scenario_number}/controversial_60_80_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                    data = pkl.load(f)

                result_df = result_df.append({'scenario': scenario_number,
                                              'dev_texts': dev_texts,
                                              'fold': fold,
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


# In[93]:


df = get_scenario_results()
df


# In[98]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Macro F1-score for each scenario (60% - 80% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.71, 0.735)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[109]:


# weighted f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Weighted F1-score for each scenario (60% - 80% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.815, 0.828)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# In[112]:


# accuracy for each scenario
accuracy_scores_s_1 = np.empty(11, dtype=np.float32)
accuracy_scores_s_2 = np.empty(11, dtype=np.float32)
accuracy_scores_s_3 = np.empty(11, dtype=np.float32)
accuracy_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

accuracy_scores = [accuracy_scores_s_1, accuracy_scores_s_2, accuracy_scores_s_3, accuracy_scores_s_4]

plt.bar(r1, [score[0] for score in accuracy_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in accuracy_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in accuracy_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in accuracy_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in accuracy_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in accuracy_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in accuracy_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in accuracy_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in accuracy_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in accuracy_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in accuracy_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Accuracy for each scenario (60% - 80% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.838, 0.846)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# <h3>Results on 80%-100% most controversial texts</h3>

# In[113]:


def get_scenario_results():
    result_df = pd.DataFrame()
    for scenario_number in [1, 2, 3, 4]:
        for dev_texts in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:

            for fold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

                with open(f'../data/partly_controversial_results/80_100/scenario_{scenario_number}/controversial_80_100_results_{scenario_number}_{dev_texts}_{fold}.pkl', 'rb') as f:
                    data = pkl.load(f)

                result_df = result_df.append({'scenario': scenario_number,
                                              'dev_texts': dev_texts,
                                              'fold': fold,
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


# In[114]:


df = get_scenario_results()
df


# In[122]:


# macro f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['macro_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Macro F1-score for each scenario (80% - 100% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Macro F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.79, 0.801)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.30)
# plt.savefig("macro_f1_each_scenario.png", dpi=300)

plt.show()


# In[127]:


# weighted f1-score for each scenario
avg_f1_scores_s_1 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_2 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_3 = np.empty(11, dtype=np.float32)
avg_f1_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    avg_f1_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()
    avg_f1_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['weighted_avg_f1-score'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

avg_f1_scores = [avg_f1_scores_s_1, avg_f1_scores_s_2, avg_f1_scores_s_3, avg_f1_scores_s_4]

plt.bar(r1, [score[0] for score in avg_f1_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in avg_f1_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in avg_f1_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in avg_f1_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in avg_f1_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in avg_f1_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in avg_f1_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in avg_f1_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in avg_f1_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in avg_f1_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in avg_f1_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Weighted F1-score for each scenario (80% - 100% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Weighted F1-score", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.902, 0.909)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()


# In[130]:


# accuracy for each scenario
accuracy_scores_s_1 = np.empty(11, dtype=np.float32)
accuracy_scores_s_2 = np.empty(11, dtype=np.float32)
accuracy_scores_s_3 = np.empty(11, dtype=np.float32)
accuracy_scores_s_4 = np.empty(11, dtype=np.float32)

for dev_texts in range(0, 21, 2):
    accuracy_scores_s_1[dev_texts // 2] = df[(df.scenario == 1) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_2[dev_texts // 2] = df[(df.scenario == 2) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_3[dev_texts // 2] = df[(df.scenario == 3) & (df.dev_texts == dev_texts)]['accuracy'].mean()
    accuracy_scores_s_4[dev_texts // 2] = df[(df.scenario == 4) & (df.dev_texts == dev_texts)]['accuracy'].mean()

width = 0.35  # the width of the bars

# Set position of bar on X axis
r1 = np.arange(1, 21, 5)

accuracy_scores = [accuracy_scores_s_1, accuracy_scores_s_2, accuracy_scores_s_3, accuracy_scores_s_4]

plt.bar(r1, [score[0] for score in accuracy_scores], width=width, edgecolor='white', label='0 dev texts')
plt.bar(r1 + 0.35, [score[1] for score in accuracy_scores], width=width, edgecolor='white', label='2 dev texts')
plt.bar(r1 + 2*0.35, [score[2] for score in accuracy_scores], width=width, edgecolor='white', label='4 dev texts')
plt.bar(r1 + 3*0.35, [score[3] for score in accuracy_scores], width=width, edgecolor='white', label='6 dev texts')
plt.bar(r1 + 4*0.35, [score[4] for score in accuracy_scores], width=width, edgecolor='white', label='8 dev texts')
plt.bar(r1 + 5*0.35, [score[5] for score in accuracy_scores], width=width, edgecolor='white', label='10 dev texts')
plt.bar(r1 + 6*0.35, [score[6] for score in accuracy_scores], width=width, edgecolor='white', label='12 dev texts')
plt.bar(r1 + 7*0.35, [score[7] for score in accuracy_scores], width=width, edgecolor='white', label='14 dev texts')
plt.bar(r1 + 8*0.35, [score[8] for score in accuracy_scores], width=width, edgecolor='white', label='16 dev texts')
plt.bar(r1 + 9*0.35, [score[9] for score in accuracy_scores], width=width, edgecolor='white', label='18 dev texts')
plt.bar(r1 + 10*0.35, [score[10] for score in accuracy_scores], width=width, edgecolor='white', label='20 dev texts')

plt.title('Accuracy for each scenario (80% - 100% most controversial texts)', fontsize=15)
plt.xlabel("Scenario", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.xticks(list(range(3, 21, 5)), ['1 - most controversial,\n random class distribution',
                                   '2 - most aggressive,\n random class distribution',
                                   '3 - most controversial,\n equal class distribution',
                                   '4 - random, \nequal class distribution'])
plt.ylim(0.91, 0.918)
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14,10)


fig.tight_layout()

plt.show()
