import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import model_selection
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

print(data['simple_journal'].value_counts())


'''Group information in columns based on mean of certain features'''
'''Comment out sections to print out the means for that particular feature'''
mean_simple_journal = data.groupby('simple_journal').mean()
#mean_issuercountrycode = data.groupby('issuercountrycode').mean()
#mean_shopperinteraction = data.groupby('shopperinteraction').mean()
# print(mean_simple_journal)
#print(mean_issuercountrycode)

'''Map binary values to each simple_journal value for classification purposes'''
'''Since it is only performance that counts, classifier is more performant with Refused mapped to 1'''
# data = data[data['simple_journal'] != 'Refused']
data['simple_journal'] = data['simple_journal'].map({'Chargeback': 1, 'Settled': 0, 'Refused': 1})

'''BarPlot  for dependent variables in this case simple_journal'''
'''Comment out print statement to see plot'''
sns.countplot(x='simple_journal', data=data, palette='hls')
# plt.show()

'''BarPlots for other varaiables, comment out print statement for plot'''
sns.countplot(x='cardverificationcodesupplied', data=data)
# plt.ylim(0, 25000)
# plt.show()

'''Check for missing values if any'''
'''comment out print statement to see missing values'''
# print(data.isnull().sum())

'''Drop columns not needed for prediction'''
'''The to be dropped columns are chosen based on count and frequency plots'''
data.drop(data.columns[[0, 1, 2, 4, 5, 7, 12, 14, 15, 16]], axis=1, inplace=True)

'''Create dummy variables for columns needed for prediction'''
data2 = pd.get_dummies(data, columns=['txvariantcode', 'currencycode', 'shopperinteraction',
                                       'cardverificationcodesupplied',
                                      'cvcresponsecode', 'accountcode'], drop_first=True)

'''HeatMap check for independence between independent variables'''
# sns.heatmap(data2.corr())
# plt.show()

print(data2.columns)
'''Split data into training and test sets'''
'''X contains all rows in data2 and all columns from 1 upwards'''
'''y contains all rows in data2 and column 0'''
X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

'''Check to make sure training set has enough data'''
print("Training set size is: {}".format(X_train.shape))

'''Logistic Regression model'''
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

'''Cross validation'''
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
result = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)

print("10-fold cross validation average accuracy: %.3f" %(result.mean()))

'''Predict test set results'''
y_pred = classifier.predict(X_test)


'''Create confusion Matrix'''
'''Add(label=np.unique(y_pred)) for non zero values of y_pred if needed '''
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

'''Print Accuracy'''
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

'''compute precision, recall, f-measure and support values'''
print(classification_report(y_test, y_pred))

'''Plot ROC curve'''

logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()

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

