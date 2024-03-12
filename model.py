import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Change these paths to ones that exist on your local machine
INPUT_PATH='input'
WORKING_PATH='working'

# Input data files are available in the read-only "input" directory
# For example, running this will list all files under the input directory

for dirname, _, filenames in os.walk(INPUT_PATH):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (WORKING_PATH) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to a temp directory, but they won't be saved outside of the current session
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

df=pd.read_csv('./input/crop-recommendation-dataset/Crop_recommendation.csv')
df.head()

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)

# from sklearn.metrics import confusion_matrix
# mat=confusion_matrix(y_test,knn.predict(X_test_scaled))
# df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
# sns.set(font_scale=1.0) # for label size
# plt.figure(figsize = (12,8))
# sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},cmap="terrain")

k_range = range(1,11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

from sklearn.svm import SVC

svc_linear = SVC(kernel = 'linear').fit(X_train_scaled, y_train)
print("Linear Kernel Accuracy: ",svc_linear.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'rbf').fit(X_train_scaled, y_train)
print("Rbf Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'poly').fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

parameters = {'C': np.logspace(-3, 2, 6).tolist(), 'gamma': np.logspace(-3, 2, 6).tolist()}
# 'degree': np.arange(0,5,1).tolist(), 'kernel':['linear','rbf','poly']

model = GridSearchCV(estimator = SVC(kernel="linear"), param_grid=parameters, n_jobs=-1, cv=4)
model.fit(X_train, y_train)

print(model.best_score_ )
print(model.best_params_ )

import pickle

# Save the trained model as a pickle string.
saved_model = pickle.dumps(model)

# Save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Save the targets dictionary to disk
with open('targets.pkl', 'wb') as f:
    pickle.dump(targets, f)

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X_test)

# Convert numerical predictions to crop labels
crop_predictions = [targets[pred] for pred in result]

print(crop_predictions)  # This will print the predicted crop labels