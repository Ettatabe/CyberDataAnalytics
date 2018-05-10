import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)



plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv("data_for_student_case.csv", header=0)
data = data.dropna()

print(data.shape)
print(list(data.columns))

# print(data['simple_journal'].value_counts())

# sns.countplot(x='simple_journal', data=data, palette='hls')
# plt.show()
# plt.savefig('counts')

#Group information in columns based on mean of certain features
mean_simple_journal = data.groupby('simple_journal').mean()
#mean_issuercountrycode = data.groupby('issuercountrycode').mean()
#mean_shopperinteraction = data.groupby('shopperinteraction').mean()
#print(mean_shopperinteraction)
#print(mean_simple_journal)
# print(mean_issuercountrycode)

# print(data['simple_journal'].unique())
# data = data[data['simple_journal'] != 'Refused']
data['simple_journal'] = data['simple_journal'].map({'Chargeback': 1, 'Settled': 0, 'Refused': 1})
# # data['cardverificationcodesupplied'] = data['cardverificationcodesupplied'].map({'TRUE': 1, 'FALSE': 0})
# data['card_id'] = data['card_id'].replace('card','11', regex=True)
# data['ip_id'] = data['ip_id'].replace('ip','11', regex=True)
# data['mail_id'] = data['mail_id'].replace('email','11', regex=True)
# del data['bookingdate']
# del data['creationdate']
# del data['cardverificationcodesupplied']
#
# data = data.replace('11NA', '11', regex=True)
# df1 = data[data['card_id'] == 'card269508']
# print(df1)
# print(data['card_id'].unique())

'''CountPlots for simple_journal'''
sns.countplot(x='simple_journal', data=data, palette='hls')
# plt.ylim(0, 600)
# plt.show()

# print(data.isnull().sum())
'''CountPlots'''
sns.countplot(x='cardverificationcodesupplied', data=data)
# plt.ylim(0, 25000)
# plt.show()

data.drop(data.columns[[0, 1, 2, 4, 5, 7, 12, 14, 15, 16]], axis=1, inplace=True)

data2 = pd.get_dummies(data, columns=['txvariantcode', 'currencycode', 'shopperinteraction',
                                       'cardverificationcodesupplied',
                                      'cvcresponsecode', 'accountcode'])

'''HeatMap'''
# sns.heatmap(data2.corr())
# plt.show()

'''SPlit intro training and test sets'''
X = data2.iloc[:,1:] #contains all rows in data2 and all columns after column 1
y = data2.iloc[:,0] #contains all rows in data2 and column 0 in data2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

'''CHeck to make sure training set has enough data'''
print(X_train.shape)

'''Logistic Reg model'''
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

'''Cross validation'''
# kfold = model_selection.KFold(n_splits=10, random_state=7)
# modelCV = LogisticRegression()
# scoring = 'accuracy'
# result = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
#
# print("10-fold cross validation average accuracy: %.3f" %(result.mean()))

'''Predict test set'''
y_pred = classifier.predict(X_test)

'''Create confusion Matrix'''
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred) #Add(label=np.unique(y_pred)) for non zero values
print(confusion_matrix)

'''Print Accuracy'''
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

'''compute precision, recall, f-measure and support'''
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

'''ROC curve'''

logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
#
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




'''Visualizations'''
# pd.crosstab(data.accountcode, data.simple_journal).plot(kind= 'bar')
# plt.title('Chargedback Frequency for Amount')
# plt.xlabel('xxx')
# plt.ylabel('Simple_journal Frequency')
# plt.ylim(0, 20000)
# plt.show()
# plt.savefig('chargeback_freq_per_currencycode')

'''histograms'''
# data.bin.hist()
# plt.title("Card Frequency for bin")
# plt.xlabel('bin')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig("Bin histogram")

