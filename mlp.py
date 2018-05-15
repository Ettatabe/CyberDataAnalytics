from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import  metrics

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from numpy import *

import pandas as pd

'''Load input file and map chargebacks to 1 and settled transactions to 0'''
data = pd.read_csv("data_for_student_case.csv")
data = data[data['simple_journal'] != 'Refused']
data['simple_journal'] = data['simple_journal'].map({'Chargeback': 1, 'Settled': 0})

'''Remove string on cardid, ipid, mailid columns since model needs floats'''
data['card_id'] = [x.strip().replace('card', '') for x in data['card_id']]
data['ip_id'] = [x.strip().replace('ip', '') for x in data['ip_id']]
data['mail_id'] = [x.strip().replace('email', '') for x in data['mail_id']]

'''Creating training, dev and test sets'''
# columns = "txid bookingdate issuercountrycode txvariantcode bin amount currencycode shoppercountrycode shopperinteraction cardverificationcodesupplied" \
#           " cvcresponsecode creationdate accountcode mail_id ip_id card_id ".split()

data = pd.get_dummies(data, columns=['txvariantcode', 'currencycode', 'shopperinteraction', 'issuercountrycode',
                                       'cardverificationcodesupplied', 'shoppercountrycode', 'currencycode',
                                      'cvcresponsecode', 'accountcode'], drop_first=True)
data.drop(data.columns[[0, 1, 5]], axis=1, inplace=True)

'''Divide data into test and training set'''
X = data.loc[:, data.columns != 'simple_journal']
y = data.loc[:, data.columns == 'simple_journal']

'''Check number of data points in minority class '''
number_fraud_records = len(data[data.simple_journal ==1])
fraud_indices = np.array(data[data.simple_journal ==1].index)

'''Pick indices of majority class'''
normal_indices = data[data.simple_journal ==0].index

'''Select random indices from the list of indices of the majority class and put them in the form of an array'''
random_normal_indices = np.random.choice(normal_indices, number_fraud_records, replace=False)
random_normal_indices = np.array(random_normal_indices)

'''Append fraud_indices and random_normal indices'''
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

'''Create undersample data from concatenated list of fraud and normal indices'''
'''Create test and training undersampling sets'''
under_sample_data = data.loc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'simple_journal']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'simple_journal']

'''Split entire dataset and the undersampled data into training and test sets'''
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)


classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)


X_train_undersample = X_train_undersample.astype(int)
X_test_undersample = X_test_undersample.astype(int)
y_train_undersample = np.nan_to_num(y_train_undersample)

# pd.isnull(np.array([np.nan, 0], dtype=float))
# pd.to_numeric(X_train, errors='coerce')

classifier.fit(X_train_undersample, y_train_undersample)

pred = classifier.predict( X_test_undersample)

score = confusion_matrix(y_test_undersample, pred)

print(score)
print(classification_report(y_test, pred))

print("Accuracy of testing dataset", metrics.accuracy_score(y_test, pred))
print("recall of  testing dataset: ", metrics.recall_score(y_test, pred))


