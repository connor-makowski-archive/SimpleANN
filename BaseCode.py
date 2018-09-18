from AnnCode import *
import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import confusion_matrix

#==============================================================================
# Data Input
#==============================================================================
input_data = r'./Data/{}'.format(filename)
data=pd.read_csv(input_data, header=[0,1,2])

#==============================================================================
# Data Manipulation Simplified
#==============================================================================
data=data.fillna(-1)
column_headers=data.columns
conversion_dict={}
conversion_dict_inv={}
cat_columns={}
output_headers=[]
tar_columns=[]
fet_columns=[]
for column_name in column_headers:
    output_headers.append(column_name[0])
    if column_name[2]=='fet':
        if column_name[1]=='cat':
            data[column_name]=data[column_name].astype('category')
            conversion_dict[column_name[0]]=dict(enumerate(data[column_name].cat.categories))
            conversion_dict_inv[column_name[0]] = {v: k for k, v in conversion_dict[column_name[0]].items()}
            data[column_name]=data[column_name].map(conversion_dict_inv[column_name[0]])
            cat_columns=pd.DataFrame(keras.utils.to_categorical(data[column_name]))
            cat_column_headers=[]
            for dict_item in cat_columns.columns:
                cat_column_headers.append(conversion_dict[column_name[0]][dict_item])
                fet_columns.append(conversion_dict[column_name[0]][dict_item])
            cat_columns.columns=cat_column_headers
            for header in cat_column_headers:
                data[header]=cat_columns[header]
        else:
            fet_columns.append(column_name[0])
    if column_name[2]=='tar':
        if column_name[1]=='cat':
            data[column_name]=data[column_name].astype('category')
            conversion_dict[column_name[0]]=dict(enumerate(data[column_name].cat.categories))
            conversion_dict_inv[column_name[0]] = {v: k for k, v in conversion_dict[column_name[0]].items()}
            data[column_name]=data[column_name].map(conversion_dict_inv[column_name[0]])
            cat_columns=pd.DataFrame(keras.utils.to_categorical(data[column_name]))
            cat_column_headers=[]
            for dict_item in cat_columns.columns:
                cat_column_headers.append(conversion_dict[column_name[0]][dict_item])
                tar_columns.append(conversion_dict[column_name[0]][dict_item])
            cat_columns.columns=cat_column_headers
            number_of_classes=len(cat_column_headers)
            for header in cat_column_headers:
                data[header]=cat_columns[header]
        else:
            tar_columns.append(column_name[0])

column_headers=data.columns
output_headers=[]
for column_name in column_headers:
    output_headers.append(column_name[0])

data.columns=output_headers
input_dimensions=len(fet_columns)
output_dimensions=len(tar_columns)

# =============================================================================
# Instantiate the model to be run
# =============================================================================

def model_to_run():
    return model_to_evaluate(input_dimensions, output_dimensions)

# =============================================================================
# Segment train and test data
# =============================================================================

x_train, x_test, y_train, y_test = train_test_split(data[fet_columns], data[tar_columns], test_size=test_pct, random_state=random_seed)

#==============================================================================
# Simple K Fold Cross Validation
#==============================================================================

if use_kfold:
    # Create an Keras Classifier
    estimator = KerasClassifier(build_fn=model_to_run, epochs=num_epochs, batch_size=batch_size, verbose=0)
    # Set K fold settings
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    # Calculate the results of a K-Fold Cross Validation
    results = cross_val_score(estimator, x_train, y_train, cv=kfold)
    # Calculate expected accuracy as average from K-Folds
    exp_accuracy=np.mean(results)

#==============================================================================
# Train and predict using model
#==============================================================================

# Instantiate Model
model = model_to_run()
# Fit Model
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
# Score the model using test data
score = model.evaluate(x_test, y_test, batch_size=batch_size)
# Output predicted probabilities
y_prob=model.predict(x=x_test, batch_size=batch_size, verbose=0, steps=None)
# Generate array of predicted class predictions
y_pred_class=y_prob.argmax(axis=-1)
# Generate array of test/actual class predictions
y_test_class=np.array(y_test).argmax(axis=-1)
#Generate confusion matrix
cm=pd.DataFrame(confusion_matrix(y_test_class, y_pred_class))



# =============================================================================
# Print Output
# =============================================================================
if use_kfold:
    print ("Expected Accuracy (kfold):", exp_accuracy)
print ("Tested Accuracy:", score[1])
print ("Confusion Matrix:")
print (cm)
