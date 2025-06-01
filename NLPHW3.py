#!/usr/bin/env python
# coding: utf-8

# <center><h1>Hemnani_Hitika_NLP_HW3</h1></center>
# <br>
# <br>

# ### Download Data

# Importing Libraries

# In[1]:


import pandas as pd


# Read the training data into a dataframe

# In[2]:


POS_training_data = pd.read_csv('data/train', sep='\t', header=None, names=['index', 'word', 'pos'])
POS_training_data.head()


# ### Vocabulary Creation

# In[3]:


###Calculating Word Occurrences in training data

vocab_count = POS_training_data['word'].value_counts().reset_index() ### to resolve a warning I had to apply reset_index()
vocab_count.columns = ['vocab', 'occurrences']
print(len(POS_training_data))


# In[4]:


##handling unknown words

##Setting Threshold as 3 for now

thresh=3
###Using a Lambda function took a long time to compute so I searched for a faster why
vocab_count['vocab'] = vocab_count['vocab'].where(vocab_count['occurrences'] >= thresh, '<unk>') ###ref:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.where.html
print(f"Total Size of the Vocabulary is {len(vocab_count)} with Threshold: {thresh}")


# In[5]:


Unknown_Count= (vocab_count['vocab'] == '<unk>').sum()
print(f"The total Occurrences of the special token ‘< unk >’ after replacement is :{Unknown_Count}")


# In[6]:


# Sorting by occurrences in descending order
vocab_count = vocab_count.sort_values(by='occurrences', ascending=False)
vocab_count.head()


# In[7]:


###Removing rare words from vocabulary as we only need one <unk> tag
vocab_count=vocab_count[vocab_count['vocab'] !='<unk>']


# In[8]:


###adding <unk> as a single row with occurences  
vocab_count = pd.concat([pd.DataFrame({'vocab': ['<unk>'], 'occurrences': [Unknown_Count]}), vocab_count])


# In[9]:


##next step we had is assigning index 
##we already have an index ...dropping that first and assigning new
vocab_count = vocab_count.reset_index(drop=True)
vocab_count = vocab_count.reset_index()
##Adding it as another column as we need in vocab file
vocab_count = vocab_count.rename(columns={'index': 'vocab_index'}) 
vocab_count.head()


# In[10]:


##Saving this Vocab_count dataframe to CSV ref:https://www.datacamp.com/tutorial/save-as-csv-pandas-dataframe?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720830&utm_adgroupid=157098107935&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=684592141898&utm_targetid=aud-1645446892440:dsa-2264919291789&utm_loc_interest_ms=&utm_loc_physical_ms=9030933&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-nam_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na-aifawfeb25&gad_source=1&gclid=CjwKCAiA2cu9BhBhEiwAft6IxGgLiXwkQUf1XpnGLlTcUHhEmyCuNp5fN-DTT5xOeN8niSbJ9qtDwBoCw_oQAvD_BwE
##As the 1st line should be <unk> we have to put header=False
vocab_count.to_csv('vocab.txt', sep='\t', columns=['vocab', 'vocab_index', 'occurrences'], index=False, header=False)


# ### Model Learning

# In[11]:


##Here as our vocabulary does not have rare words ... as far as I know we should remove those words from training data as well
thresh = 3 
##POS_training_data['word'] = POS_training_data['word'].apply(lambda x: x if x in vocab_count[vocab_count['occurrences'] >= thresh]['vocab'].values else '<unk>')
##aabove function is taking a lot of time to run so to optimize it I will use the method I used in vocabulary
##set of words above threshold
valid_vocab = vocab_count[vocab_count['occurrences'] >= thresh]['vocab'].values

# Step 2: Replace words in POS_training_data with '<unk>' if not in valid vocab
POS_training_data['word'] = POS_training_data['word'].where(POS_training_data['word'].isin(valid_vocab), '<unk>')


# In[12]:


POS_training_data.head()


# ## HMM

# In[13]:


##seperating words and POS Togs
word = POS_training_data['word'].tolist()
pos = POS_training_data['pos'].tolist()


# In[14]:


unique_word = set(word)
unique_pos = set(pos)
##we want to work with numbers to help in working with numbers instead of text.
num_word = {w: i for i, w in enumerate(unique_word)}
num_pos = {p: i for i, p in enumerate(unique_pos)}


# In[15]:


###Creating Dictionary for Transition parameter and Emision parameter
from collections import Counter,defaultdict

# ##Counting Occurences and transitions
# tparams_counts = defaultdict(lambda: defaultdict(int)) 
# pos_counts = defaultdict(int) 

# ##previous pos_tag and currect pos tag ...how ofter one appears after another
# for i in range(1, len(pos)):
#     prev_pos = pos[i - 1]
#     curr_pos = pos[i]

#     tparams_counts[prev_pos][curr_pos] += 1  
#     pos_counts[prev_pos] += 1 

# ##computing Probability on the basis of count
# tparams_probs = defaultdict(dict)

# for prev_pos, next_pos in tparams_counts.items():
#     total_count = pos_counts[prev_pos] 
    
#     for next_pos, count in next_pos.items():
#         tparams_probs[prev_pos][next_pos] = count / total_count 


##I thought of a new simpler solution 
tparams_counts = Counter(zip(pos[:-1], pos[1:]))

# Count total occurrences of each POS tag as a previous tag
pos_counts = Counter(pos)

# Compute transition probabilities: P(current_tag | previous_tag)
tparams_probs = defaultdict(dict)

for (prev_pos, curr_pos), count in tparams_counts.items():
    tparams_probs[prev_pos][curr_pos] = count / pos_counts[prev_pos]


# In[16]:


## For Emission Probability
# Counting how many times each (POS tag, word) appears and Total Occurences of each tag  
eparams_counts = Counter(zip(pos, word))
pos_counts = Counter(pos)

##computing probability on the basis of count
eparams_probs = defaultdict(dict)

for (p, w), count in eparams_counts.items():
    eparams_probs[p][w] = count / pos_counts[p]


# In[17]:


#print("Transition Probabilities:", tparams_probs)
#print("Emission Probabilities:", eparams_probs)


# In[18]:


print(type(POS_training_data['index']))


# In[19]:


##Computing Probility of Initial word
initial_counts = Counter()

# Counting which POS tags comes in the start of sentence
# for data in POS_training_data:
#     if data['index'].astype('int') == 1:
#         initial_counts[data['pos']] += 1
for index, row in POS_training_data.iterrows():
    if row['index'] == 1:
        initial_counts[row['pos']] += 1

##computing probability on the basis of count
total_starts = sum(initial_counts.values())  # Total number of first words in all sentences
initial_probs = {tag: count / total_starts for tag, count in initial_counts.items()}

# Output the initial probabilities
#print("Initial Probabilities:", initial_probs)


# In[36]:


### 4. Computing Number of Parameters
num_transition_params = len(tparams_counts)  # Number of unique (s, s') pairs
num_emission_params = len(eparams_counts)  # Number of unique (s, x) pairs

### Print Results ###
print(f"Number of transition parameters: {num_transition_params}")
print(f"Number of emission parameters: {num_emission_params}")


# In[20]:


##Saving HMM model
import json

model = {
    "transition": tparams_probs,  
    "emission": eparams_probs,    
    "initial": initial_probs      
}

with open("hmm.json", "w") as f:
    json.dump(model, f, indent=4) 

print("Model saved to hmm.json")


# ### Greedy Decoding with HMM 

# In[21]:


dev_data = pd.read_csv('data/dev', sep='\t', header=None)
dev_data.columns = ['index', 'word', 'pos']


# In[22]:


dev_data.head()


# In[23]:


# The development data is a sequence of words with their POS tags. We need to group the data into sentences. A new sentence starts when the index column is 1
###The previous sentence ends when a new sentence starts (i.e., when we encounter the next row with an index == 1)
dev_sentences = []
current_sentence = []

for _, row in dev_data.iterrows():
    if row['index'] == 1 and current_sentence:
        dev_sentences.append(current_sentence)
        current_sentence = []
    current_sentence.append(row)

dev_sentences.append(current_sentence)


# In[24]:


#print(sentences[1])


# In[25]:


###Applying Greedy Algorithm

dev_predictions = []
# For each sentence the words are extracted
for sentence in dev_sentences:
    words = [row['word'] for row in sentence]
    predicted_pos = []

    for word in words:
        best_pos = 'NN'  # Default POS tag if word is unknown
        best_prob = 0

        # For each word we search through all possible POS tags and pick the one with the highest emission probability
        for pos in eparams_probs:
            if word in eparams_probs[pos]:
                prob = eparams_probs[pos][word]
                if prob > best_prob:
                    best_prob = prob
                    best_pos = pos

        predicted_pos.append(best_pos)

    # The predictions are stored in the same format as the original input data with the word and the predicted POS tag
    for i, row in enumerate(sentence):
        dev_predictions.append((row['index'], row['word'], predicted_pos[i]))


# In[26]:


# ##ref:https://www.w3schools.com/python/python_file_write.asp
with open('greedy_dev.out', 'w') as f:
    for pred in dev_predictions:
        f.write(f"{pred[0]}\t{pred[1]}\t{pred[2]}\n")

print("Development predictions saved to greedy_dev.out")


# In[27]:


###comploted running eval.py on test data 
###output:total: 131768, correct: 114962, accuracy: 87.25%

test_data = pd.read_csv('data/test', sep='\t', header=None, names=['index', 'word', 'pos'])


# In[28]:


## Grouping the test data into sentences like we did before for training data
test_sentences = []
current_sentence = []

for _, row in test_data.iterrows():  # Fix: Use test_data instead of dev_data
    if row['index'] == 1 and current_sentence:
        test_sentences.append(current_sentence)
        current_sentence = []
    current_sentence.append(row)

test_sentences.append(current_sentence)


# In[29]:


##Performing Greedy algorithm similarly for test data
test_predictions = []

for sentence in test_sentences:
    words = [row['word'] for row in sentence]
    predicted_pos = []

    for word in words:
        best_pos = 'NN'  # Default POS tag if word is unknown
        best_prob = 0

        # Find the POS tag with the highest emission probability for the current word
        for pos in eparams_probs:
            if word in eparams_probs[pos]:
                prob = eparams_probs[pos][word]
                if prob > best_prob:
                    best_prob = prob
                    best_pos = pos

        predicted_pos.append(best_pos)

    for i, row in enumerate(sentence):
        test_predictions.append((row['index'], row['word'], predicted_pos[i]))


# In[30]:


with open('greedy.out', 'w') as f:
    for pred in test_predictions:
        f.write(f"{pred[0]}\t{pred[1]}\t{pred[2]}\n")

print("Predictions saved to greedy.out")


# ### Viterbi Decoding with HMM

# In[31]:


def viterbi_decoding(sentence, transition_paramter, emission_paramter, initial_paramter, tags):
    num_words = len(sentence)
    num_tags = len(tags)

    viterbi = [{} for _ in range(num_words)]  
    backpointer = [{} for _ in range(num_words)]  # for previous tag
    # Smoothing value for unknown words, initial probabilities, and transitions to increase accuracy
    # I got 47% accuracy without using smoothing as it fails to take unknown values and make prob 0
    smoothing_value = 1e-10
    #  for first word using initial_probability
    first_word = sentence[0]
    for tag in tags:
        initial_prob = initial_paramter.get(tag, smoothing_value)
        if first_word in emission_paramter[tag]:
            emission_prob = emission_paramter[tag][first_word]
        else:
            emission_prob = smoothing_value

        viterbi[0][tag] = initial_prob * emission_prob
        backpointer[0][tag] = None  # first tag would not have previous tag

    # from next word
    for t in range(1, num_words):
        word = sentence[t]
        for current_tag in tags:
            max_prob = -1
            best_prev_tag = None

            for prev_tag in tags:
                if word in emission_paramter[current_tag]:
                    emission_prob = emission_paramter[current_tag][word]
                else:
                    emission_prob = smoothing_value
                ###Using Formula viterbi[t][current_tag]=  max of previous tag (viterbi[t−1][prev_tag]×t(current_tag∣prev_tag)×e(word∣current_tag))
                ###ref for this formula:https://www.geeksforgeeks.org/viterbi-algorithm-for-hidden-markov-models-hmms/
                transition_prob = transition_paramter.get(prev_tag, {}).get(current_tag, smoothing_value)

                # Calculate the probability
                prob = viterbi[t - 1][prev_tag] * transition_prob * emission_prob
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag

            viterbi[t][current_tag] = max_prob
            backpointer[t][current_tag] = best_prev_tag

    # Maximum Probability Selected 
    best_last_tag = max(viterbi[-1], key=viterbi[-1].get)
    best_path = [best_last_tag]
    
    for t in range(num_words - 1, 0, -1):
        best_last_tag = backpointer[t][best_last_tag]
        best_path.insert(0, best_last_tag)

    return best_path


# In[32]:


##same as greedy...
tags = list(eparams_probs.keys())

viterbi_dev_predictions = []

for sentence in dev_sentences:
    words = [row['word'] for row in sentence]
    predicted_tags = viterbi_decoding(words, tparams_probs, eparams_probs, initial_probs, tags)

    for i, row in enumerate(sentence):
        viterbi_dev_predictions.append((row['index'], row['word'], predicted_tags[i]))


# In[33]:


with open('viterbi_dev.out', 'w') as f:
    for pred in viterbi_dev_predictions:
        f.write(f"{pred[0]}\t{pred[1]}\t{pred[2]}\n")

print("Viterbi predictions saved to viterbi_dev.out")


# In[34]:


##For test data
test_viterbi_predictions = []

for sentence in test_sentences:
    words = [row['word'] for row in sentence]
    predicted_tags = viterbi_decoding(words, tparams_probs, eparams_probs, initial_probs, tags)

    for i, row in enumerate(sentence):
        test_viterbi_predictions.append((row['index'], row['word'], predicted_tags[i]))


# In[35]:


with open('viterbi.out', 'w') as  f:
    for pred in test_viterbi_predictions:
        f.write(f"{pred[0]}\t{pred[1]}\t{pred[2]}\n")

print("Viterbi predictions for test data saved to viterbi.out")


# In[ ]:





# In[ ]:




