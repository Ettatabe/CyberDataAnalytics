from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from numpy import *
import numpy as np

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
data.drop(data.columns[[ 1, 5]], axis=1, inplace=True)

X = pd.DataFrame.as_matrix(data)
y = data.simple_journal
y = y.reshape(y.shape[0],1)

'''Split entire dataset and the undersampled data into training and test sets'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.06)

'''Classifier'''
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)


X_train = X_train.astype(int)
X_test= X_test.astype(int)

classifier.fit(X_train, y_train)



pred = classifier.predict( X_test)

confusionmatrix_undersample = confusion_matrix(y_test, pred)


print(confusionmatrix_undersample)
print(classification_report(y_test, pred))


print("Accuracy of testing dataset", metrics.accuracy_score(y_test, pred))

print("recall of  testing dataset: ", metrics.recall_score(y_test, pred))



