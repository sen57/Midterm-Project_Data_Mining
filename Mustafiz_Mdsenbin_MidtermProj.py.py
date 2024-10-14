#!/usr/bin/env python
# coding: utf-8

# In[67]:


#%pip install pandas
#%pip install apriori_python
#%pip install mlxtend
import pandas as pd
from itertools import combinations
from apriori_python.apriori import apriori
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time


# In[68]:


##support calculation


# In[69]:


def calculate_support(data):
    item_dict = {}
    num_transaction = len(data)

    # doing Iteratation with each transaction and count all the items
    for transaction in data['transaction']:
        items = transaction.split(',')  #split where is comma
        for item in items:
            item = item.strip()
            if item in item_dict:
                item_dict[item] += 1  #counting
            else:
                item_dict[item] = 1

    # support calculation for each item
    support = {}
    for item, count in item_dict.items():
        support[item] = count / num_transaction

    return support


# In[70]:


## creating combination and make sure hey are below the minimum support value


# In[71]:


def parse_transaction(transaction):
    res = set()  #blank ser
    for item in transaction.split(','):
        res.add(item.strip())   #add items in res set
    return res


# In[72]:


def generate_combinations_and_calculate_support(data, freq_items, support):
    #calling the transaction data first
    # then set of parse_transaction will convert into list
    transactions = data['transaction'].apply(parse_transaction).tolist()  
    num_transaction = len(transactions)

    k = 2  # Start with 2-item combinations
    comb_sup = {}   #emply dictionary for combination 
    
    while len(freq_items) > 1:
        combs = list(combinations(freq_items, k))
        present_comb_sup = {}

        # # support calculation for each combination
        for comb in combs:                 
            summation = 0
            for transaction in transactions:
                if set(comb).issubset(transaction):  #if subset then go forward for summation
                    summation += 1
            sup = summation / num_transaction
            if sup >= support:
                present_comb_sup[comb] = sup

        # the combinations with minimum support, need to  store them and filter the itemset
        if present_comb_sup:
            comb_sup.update(present_comb_sup)
            freq_items = set()
            for comb in present_comb_sup.keys():
                for item in comb:
                    freq_items.add(item)
        else:
            break

        # Stop the code if only one combination is left with minimum support
        if len(freq_items) == 1:
            print(f"Only one combination is left with support ≥ {support}. Stopping iteration.")

        k += 1  # Move to the next combination size

    return comb_sup


# In[73]:


## for the combination which pass the minimum threshold support, will go forward for confidence calculation


# In[74]:


def confidence_calc(comb_sup, sup_vals, support, confidence):
    confidence_dict = {}  #empty dicttionary for confidence

    for comb, both_supp in comb_sup.items():
        for r in range(1, len(comb)):
            for prev in combinations(comb, r): # calculationg the combinations of 'r' items
                next = tuple(set(comb) - set(prev)) # tupple so it will not change in this loop

                if len(prev) == 1:
                    prev_sup = sup_vals[prev[0]] # first single support value
                else:
                    prev_sup = comb_sup.get(prev, 0) # the combination supprot

                # it will Calculate the confidence only if previous support is ok and more than minimum
                if prev_sup >= support:
                    
                    conf = both_supp / prev_sup 
                    if conf >= confidence:
                        confidence_dict[(prev, next)] = (conf, both_supp)

    return confidence_dict


# In[75]:


#checking of values over threshold values and print as requirement


# In[76]:


def calculate_support_and_confidence(ssd, support, confidence):
    
    # Start count start position
    start_time = time.time()
    for t in range(1000000):
        pass
    data = ssd  #take the data inside this main function
    
    # Calculate individual item support
    support_values = calculate_support(data)   
    
    # Filter items with support more than minimum 
    filtered_itemset = []
    for item in support_values:
        if support_values[item] >= support:
            filtered_itemset.append(item)
    # If no item has minimum support, return
    if not filtered_itemset:
        print(f"No items with support ≥ {support}") #sup
        return
    
    # Generate combinations and calculate support for each
    combination_support_values = generate_combinations_and_calculate_support(data, filtered_itemset, support)
    
    # Calculate confidence for itemsets with minumum support 
    confidence_values = confidence_calc(combination_support_values, support_values, support, confidence)    
    print("\n Confidence and Support values for Brute Force process \n")
   
    ##print function for desired output format as i have my output in tuple format
    ##so at first i make it list, then the list to string which i will print
    rule = 1
    for (prev, next), (conf, sup) in confidence_values.items():
        # Unpack any single-item tuples in `prev` and `next` for proper display
        prev_items = []
        for item in prev:
            if isinstance(item, tuple):
                prev_items.append(item[0])
            else:
                prev_items.append(item)
        # now the next item
        if isinstance(next, tuple):
            next_item = next[0]
        else:
            next_item = next
        # Display the frequent itemset as a set
        items = prev_items + [next_item]
        item_string = ""
        for item in items:
            if item_string == "":
                item_string = str(item)  # First item, no comma
            else:
                item_string += ", " + str(item)  # Add comma for the rest
        
        print("Freq. Itemset {" + item_string + "}")
        
        # print the rule with the rule number
        print(f"Rule {rule}: {[str(item) for item in prev_items]} -> [{str(next_item)}]")
        
        # Print support and confidence values
        print(f"Support Count: {sup:.2f}")
        print(f"Confidence: {conf:.2f}")
        print("-" * 30)
        # Increment the rule number for the next rule
        rule += 1
   
    # time count complete
    end_time = time.time()
    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time for Brute Force: {execution_time} seconds")


# In[77]:


## asking for input..there are 5 dataset..we have to choose 1 for finding out association rules


# In[78]:


transaction_sets = {
                    1: 'book.csv',
                    2: 'colors.csv',
                    3: 'flower.csv',
                    4: 'games.csv',
                    5: 'tree.csv'
                    }
print("Welcome to Apriori 2.0!")
print("User please select your store:")
print("1. Book")
print("2. Colors")
print("3. Flower")
print("4. Games")
print("5. Tree")

number = int(input("Please enter a number between 1 and 5: "))   #taking input

while number < 1 or number > 5:
    number = int(input("Please enter a number between 1 and 5: "))  #if the input is not in between 1-5 it will run forever, infinite loop

df = pd.read_csv(transaction_sets[number])

ssd=df

# asking for threshold support and confidence values in percentage
support1 = int(input("Enter minimum support (%) values (1-100): "))
support=(support1/100)
confidence1 = int(input("\nEnter minimum confidence (%) values (1-100): "))
confidence=(confidence1/100)
calculate_support_and_confidence(ssd, support, confidence)


# In[79]:


## using default Apriori Algorithm


# In[80]:


print ("Apriori Algorithm\n")
# Specify minSup and minConf values
start_time = time.time()
for t in range(1000000):
    pass

minSup = support;
minConf = confidence;
data=ssd
transactions = data['transaction'].apply(parse_transaction).tolist()
num_transaction = len(transactions)

# Run Apriori algorithm
freqItemSet, rules = apriori(transactions, minSup=minSup, minConf=minConf)

# Print the associtions rule
for i, rule in enumerate(rules): 
    #to add list and tupple we use enumerate
    #convert and print the value as a string
    rule_str = f"Rule {i + 1}: {rule[0]} -> {rule[1]}\nConfidence : {rule[2]:.2f}"
    print(rule_str)
    print("-" * 30)  #just 30 times space

# End time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print(f"Execution Time Apriori Algorithm: {execution_time} seconds")


# In[81]:


## using default FP Growth algoritm


# In[82]:


print('NOW FP growth')

#Preprocess the data for FP-Growth (use TransactionEncoder to convert data into a DataFrame)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

#Start timing
start_time = time.time()
for t in range(1000000):
    pass

#Run FP-Growth algorithm
frequent_itemsets_fp = fpgrowth(df, min_support=minSup, use_colnames=True)

#Generate association rules from the frequent itemsets
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=minConf)

#Print the rules 
print("Association Rules\n")
for i, row in rules_fp.iterrows():
    prev = ', '.join(list(row['antecedents']))
    later = ', '.join(list(row['consequents']))
    support_value = row['support']  # Get support from the rule
    #print the output
    print(f"Rule {i + 1}: {{{prev}}} -> {{{later}}},\nSupport: {support_value:.2f}")
    print("-" * 30)

#End time
end_time = time.time()

#Calculate execution time
execution_time = end_time - start_time
print(f"\nExecution Time: FP-Growth {execution_time} seconds")


# In[ ]:




