'''Hi there! My name is Reza Javadzadeh. Here is the whole codes of Machine Learning with Scikit-Learn module course
which is taught in https://www.koolac.org. You can purchase this course from https://koolac.org/product/machine-learning/
For more information like how to use this code or more projects, check out my Github account.

-- -- Github: https://github.com/Reza-Javadzadeh
-- -- LinkedIn: https://linkedin.com/in/reza-javadzadeh'''




'''P01-01-Terminology + Reading Data'''


## 03

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# #######################
# ## Reading File
# #######################
# ## Read and displaying:
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info())
# print(df.head())
#
# ## Defining x and y (input and output):
# x=df.iloc[:,3:-1].values # we use numeric feature yet, so we didn't add Gender and Type columns. For make it more simple in the process, we add ".values" method to turn it as numpy array.
# y=df.iloc[:,-1].values # 'Purchase' column as target value or label. For make it more simple in the process, we add ".values" method to turn it as numpy array.
#
# print('x:\n',x,end='\n\n')
# print('y:\n',y,end='\n\n')
#

'''P01-02-Scaling'''

# ##01
#
# ###############################################
# ## preprocessing
# ###############################################
#
# ## scaling:
# ##02 MinMaxScaler
#
# from sklearn.preprocessing import MinMaxScaler
#
# scaler=MinMaxScaler()  # we bulit an object for doing MinMax scaling by MinMaxScaler class. the feature_scaling default value is in range [0,1], we can change it as our desire.
# xx=scaler.fit_transform(x)
# yy=scaler.fit_transform(y.reshape(-1,1)) # it should be 2D array , 1D array return error. so we must reshape it like: array.reshape(-1,1)
# print('x after preprocessing (i.e. MinMax scaling):\n',xx,end='\n\n')
# print('y after preprocessing (i.e. MinMax scaling):\n',yy,end='\n\n')
#
# # scaler=MinMaxScaler(feature_range=(2,5))  # we bulit an object for doing MinMax scaling by MinMaxScaler class. the feature_scaling has been changed to [2,5] interval. note that it should be in tuple form.
# # xx=scaler.fit_transform(x)
# # yy=scaler.fit_transform(y.reshape(-1,1)) # it should be 2D array , 1D array return error. so we must reshape it like: array.reshape(-1,1)
# # print('x after preprocessing (i.e. MinMax scaling):\n',xx,end='\n\n')
# # print('y after preprocessing (i.e. MinMax scaling):\n',yy,end='\n\n')
#
# ## 05  StandardScaler
#
# from sklearn.preprocessing import StandardScaler
#
# scaler=StandardScaler()  # we bulit an object for doing Standard scaling by MinMaxScaler class.
# xx=scaler.fit_transform(x)
# yy=scaler.fit_transform(y.reshape(-1,1)) # it should be 2D array , 1D array return error. so we must reshape it like: array.reshape(-1,1)
# print('x after preprocessing (i.e. Standard scaling):\n',xx,end='\n\n')
# print('y after preprocessing (i.e. Standard scaling):\n',yy,end='\n\n')
#
#
# ## 07 ZScore
#
# ## if we tend to compute StandardScaling for whole society (i.e.: the complete sample space and ddof=0) , we utilize "sklearn.preprocessing.StandardScaling".
# ## else , (i.e.: apply it for some sample and ddof=1), we should utilize "scipy.stats.zscore". sklearn doesn't support ddof > 0 .
#
# from scipy.stats import zscore
#
# xx=zscore(x,ddof=1) # the default ddof is equaled to 0. so we should change it to ddof=1
# yy=zscore(y.reshape(-1,1),ddof=1)
# print('x after preprocessing (i.e. zscore):\n',xx,end='\n\n')
# print('y after preprocessing (i.e. zscore):\n',yy,end='\n\n')

# ##8 Scaling or Dataset
#
# # so as epilogue of this lesson, let's do what we have untill now all over again:
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# #######################
# ## Reading File
# #######################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# ## Read and displaying:
# print(df.info())
# print(df.head())
#
# ## Defining x and y (input and output):
# x=df.iloc[:,3:-1].values # we use numeric feature yet, so we didn't add Gender and Type columns. For make it more simple in the process, we add ".values" method to turn it as numpy array.
# y=df.iloc[:,-1].values # 'Purchase' column as target value or label. For make it more simple in the process, we add ".values" method to turn it as numpy array.
#
# print('x:\n',x,end='\n\n')
# print('y:\n',y,end='\n\n')
#
# # ###############################################
# # ## preprocessing
# # ###############################################
# #
# # ## scaling:
#
# from sklearn.preprocessing import MinMaxScaler,StandardScaler
# from scipy.stats import zscore
#
# scalerMINMAX=MinMaxScaler() #feature_range set to (0,1) as default
# scalerSTANDARD=StandardScaler()
#
# xx=scalerMINMAX.fit_transform(x)
# yy=scalerMINMAX.fit_transform(y.reshape(-1,1))
#
# print('x after preprocessing (i.e. MinMax scaling):\n',xx,end='\n\n')
# print('y after preprocessing (i.e. MinMax scaling):\n',yy,end='\n\n')
#
# xx=scalerSTANDARD.fit_transform(x)
# yy=scalerSTANDARD.fit_transform(y.reshape(-1,1))
#
# print('x after preprocessing (i.e. Standard scaling):\n',xx,end='\n\n')
# print('y after preprocessing (i.e. Standard scaling):\n',yy,end='\n\n')
#
# xx=zscore(x,ddof=1)
# yy=zscore(y.reshape(-1,1),ddof=1)
#
# print('x after preprocessing (i.e. zscore and ddof==1):\n',xx,end='\n\n')
# print('y after preprocessing (i.e. zscore and ddof==1):\n',yy,end='\n\n')

'''P02-01-KNN:'''

# ## 01 we are going to use K Nearest Neighbors model for learing in this section:
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import StandardScaler # we wanna scale our model as Standard one.
# from sklearn.neighbors import KNeighborsClassifier # our model is KNN in this chapter.
#
#
# ##################
# ## Reading Data:
# ##################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n\n',df.head(),end='\n\n')
#
# ## Defining x and y as input and output:
#
# x=df.iloc[:,3:-1].values
# y=df.iloc[:,-1].values.reshape(-1,1)
#
# ##################
# ## Preprocessing:
# ##################
#
# ## Scaling with Standard shape:
# scaler=StandardScaler()
# x=scaler.fit_transform(x)
# # y=scaler.fit_transform(y)  ## Seems the output shouldn't get normalize, otherwise return error! [ValueError: It shouldn't be 'countinious'.]
#
# ##################
# ## Building the model:
# ##################
#
# model=KNeighborsClassifier(n_neighbors=4)
# model.fit(x,y) ## Python Intepreter also return a Warning. [DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().]



'''P02-02-Prediction and Evaluation-Part 01'''

# ## 02 train test split-Python
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# #############
# ## reading data:
# #############
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head: \n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1].values #output
#
#
# ###############################
# ## preprocessing:
# ###############################
#
# ## train test split (from model selection):
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25) ## Note that we didn't stratify our data during splitting data to train and test.
#
# ## scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test) # '''We should only use transform(),because the TEST data should only transform from pattern of train data which has been trained and fitted.'''
#
# print(x_train.shape)
# print(x_test.shape)
#
# ###################
# ## Building the model:
# ###################
#
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=4)
# model.fit(x_train,y_train)



# ## 03 stratify:
#
# ## This time , we will keep stratify condition when we tend to split our data to test and train w.r.t. Y.
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# #############
# ## reading data:
# #############
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head: \n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1].values #output
#
#
# ###############################
# ## preprocessing:
# ###############################
#
# ## train test split:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y) # we stratify our data during splitting
# print('x_train "before scaling"\n',x_train)
# print('x_test "before scaling"\n',x_test)
#
# ## scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)  # '''We should only use transform(),because the TEST data should only transform from pattern of train data which has been trained and fitted.'''
# print('x_train "after scaling"\n',x_train)
# print('x_test "after scaling"\n',x_test)
#
# #########################
# ## Building the model
# #########################
#
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=5)
# model.fit(x_train,y_train)

## 04    train test split (complementary)


'''NOTE : this video is about difference between "fit_transform" for trained data and "transform" for test data,which has been commented beside of codes in previous lesson. '''


# ## 05-prediction and accuracy (accuracy is just one of the Evaluation metrics )
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ########################
# ## Reading Data
# ########################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1].values #output
#
# ##################################
# ## Preprocessing:
# ##################################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40) # we stratifed our data and put seed==40 , as make it same with koolac.
#
# ## scaling:
# from sklearn.preprocessing import StandardScaler #Going to choose Standard normalization as our scaler
#
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ################################
# ## Building the model:
# ################################
# from sklearn.neighbors import KNeighborsClassifier # utilizing KNN as our model to training the data
#
# model=KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
#
# ##################################
# ## Predicting and Evaluating:
# ##################################
#
# ## predicting:
# y_pred=model.predict(x_test)
#
# ## evaluating:
# from sklearn.metrics import accuracy_score # we use Accuracy metric to evaluate our model.
# ## ---- accuracy:
# acc=accuracy_score(y_test,y_pred)
# print(f'Our accuracy is {acc*100}%.')

## 06 accuracy shortcomings:

'''In this video we can conclude that the ACCURACY as an evaluating index, is not proper for biased outputed data (i.e.: oue labels,"y"). 
for example it's suitable for about 50%-50% unbiased output, not 90%-10% !!!'''


## 07 confusion matrix-concept:

'''Since may accuracy metric depends on unbiased or biased output have different efficiency, we should introduce and work with 
another evaluation indices. one of them call "Confusion Matrix". '''


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ########################
# ## Reading Data
# ########################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1].values #output
#
# ##################################
# ## Preprocessing:
# ##################################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40) # we stratifed our data and put seed==40 , as make it same with koolac.
#
# ## scaling:
# from sklearn.preprocessing import StandardScaler #Going to choose Standard normalization as our scaler
#
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ################################
# ## Building the model:
# ################################
# from sklearn.neighbors import KNeighborsClassifier # utilizing KNN as our model to training the data
#
# model=KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
#
# ##################################
# ## Predicting and Evaluating:
# ##################################
#
# ## predicting:
# y_pred=model.predict(x_test)
#
# ## evaluating:
#
# ## ---- accuracy:
# from sklearn.metrics import accuracy_score # we use Accuracy metric to evaluate our model.
# acc=accuracy_score(y_test,y_pred)
# print(f'Our accuracy is {acc*100}%.',end='\n\n')
# ## ------ confusion matrix:
# from sklearn.metrics import confusion_matrix
# label_orders=[0,1]
# cm=confusion_matrix(y_test,y_pred,labels=label_orders) #note that the default labels for CM is np.sort(<<<<y_test>>>>> or <<<<<y_pred>>>>>>) i.e.: it's ascending.which is equal to [0,1] here, so it wasn't nececery to mention that in code.
# print('The Confusion Matrix:\n',cm,end='\n\n')
#
# ## 09 confusion matrix-in data frame format
#
# '''We can use a Pandas DataFrame to visualize our confusion matrix better.
# So : '''
# labels_order=[0,1]
# cm_df=pd.DataFrame(cm,index=labels_order,columns=labels_order)
# print('The Confusion Matrix Dataframe:\n',cm_df,end='\n\n')


## 10 normalized confusion matrix

'''Confusion matrix has better comprehension to the users when describes as Normalized one. So we introduce Normalize Confusion Matrix:'''

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ######################
# ## Reading Data:
# ######################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# ##################################
# ## Preprocessing:
# ##################################
#
# ## train test splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40)
#
# ## scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ################################
# ## Building model:
# ################################
#
# ## KNN:
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
#
#
# ################################
# ## Predicting and Evaluating:
# ################################
#
# ##predicting:
# y_pred=model.predict(x_test)
#
# ##evaluating:
#
# ## ---- Accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_test,y_pred)
# print('The accuracy is {}%'.format(acc*100))
#
# ## ---- confusion matrix:
# from sklearn.metrics import confusion_matrix
# labels_order=[0,1]
# cm=confusion_matrix(y_test,y_pred,labels=labels_order)
# cm_df=pd.DataFrame(cm,index=labels_order,columns=labels_order)
# print('Confusion Matrix DF:\n',cm_df,end='\n\n')
#
# ## ---- Normalize Confusion Matrix:
# normalized_cm=np.round(cm/np.sum(cm,axis=1).reshape(-1,1),2)
# normalized_cm_df=pd.DataFrame(normalized_cm,index=labels_order,columns=labels_order)
# print('Normalized Confusion Matrix DF:\n',normalized_cm_df,end='\n\n')


## 11 heatmap for confusion matrix:

'''Now we are going to plot our (Normalized)CM. We should use Seaborn module beside matplotlib. the function that we use is heatmap,
i.e: sns.heatmap().'''




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ######################
# ## Reading Data:
# ######################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# ##################################
# ## Preprocessing:
# ##################################
#
# ## train test splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40)
#
# ## scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ################################
# ## Building model:
# ################################
#
# ## KNN:
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
#
#
# ################################
# ## Predicting and Evaluating:
# ################################
#
# ##predicting:
# y_pred=model.predict(x_test)
#
# ##evaluating:
#
# ## ---- Accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_test,y_pred)
# print('The accuracy is {}%'.format(acc*100))
#
# ## ---- confusion matrix:
# from sklearn.metrics import confusion_matrix
# labels_order=[0,1]
# cm=confusion_matrix(y_test,y_pred,labels=labels_order)
# cm_df=pd.DataFrame(cm,index=labels_order,columns=labels_order)
# print('Confusion Matrix DF:\n',cm_df,end='\n\n')
#
# ## ---- Normalized Confusion Matrix:
# normalized_cm=np.round(cm/np.sum(cm,axis=1).reshape(-1,1),2)
# normalized_cm_df=pd.DataFrame(normalized_cm,index=labels_order,columns=labels_order)
# print('Normalized Confusion Matrix DF:\n',normalized_cm_df,end='\n\n')
#
# ## ---- ---- confusion matrix heatmap:
# import seaborn as sns
#
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=labels_order,yticklabels=labels_order,cbar_kws={'label':'Color Bar','orientation':'vertical'}) #we can also put cm_df(i.e: a Dataframe).
# # "annot" means the numbers must show in middle of heatmap. "fmt" means how many digits after floating point should be previewed.
# # for instance, fmt='0.2f' implies that 2 number of floating and it should be FIXED.(i.e: also the number was an integer it should preview the rest of it
# # Exmpl: 34.00). we can use "cbar_kws" as configuring the color bar.the color bar default orientation is vertical.
# plt.title('Normalized Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
#
# ## ---- ---- normalized confusion matrix heatmap:
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=labels_order,yticklabels=labels_order,cbar_kws={'label':'Color Bar','orientation':'vertical'}) #we can also put cm_df
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()






'''P02-02-Prediction and Evaluation-Part 02'''

## 12 recall, precision, specificity-theory

'''in this video, explained about Recall(Sensitivity) and Precision as important indices which has Plug&Play function in sklearn. the another index, the 
 Specification doesn't have a function or module in python. so we should compute it with our handwritten code.'''


## 13 recall, precision, specificity-Python

'''let's apply these indices to our confusion matrix.'''

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ################################
# ## Reading Data:
# ################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# #########################
# ## Preprocessing:
# #########################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40)
#
# ## Scaling:
# ## -- Standard Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ########################
# ## Building our model:
# ########################
# ## KNN:
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
#
# ########################
# ## Prediction and Evaluation:
# ########################
#
# ## Prediction:
# y_pred=model.predict(x_test)
#
# ## Evaluation:
# ## ---- accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_test,y_pred)
# print(f'The Accuracy is {acc*100}%',end='\n\n')
# ## ---- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# labels_order=[0,1]
# cm=confusion_matrix(y_test,y_pred,labels=labels_order)
# cm_df=pd.DataFrame(cm,index=labels_order,columns=labels_order)
# print('CM DF: \n',cm_df,end='\n\n')
#
#
# ## ---- ---- Confusion Matrix Recall(Sensitivity),Precision and Specificity:
# ## Recall:
# from sklearn.metrics import recall_score
# recall=recall_score(y_test,y_pred,labels=labels_order)
# print('Recall: ',recall,end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_test,y_pred,labels=labels_order)
# print('Precision: ',precision,end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])
# print('Specificity: ',specificity,end='\n\n')
#
#
# ## ---- ---- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm_df,cmap='Greens',fmt='0.2f',annot=True,xticklabels=labels_order,yticklabels=labels_order,cbar_kws={'label':'Color Bar',"orientation":"vertical"})
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.tight_layout()
#
#
# ## ---- Normalized Confusion Matrix:
# normalized_cm=cm/np.sum(cm,axis=1).reshape(-1,1)
# normalized_cm_df=pd.DataFrame(normalized_cm,index=labels_order,columns=labels_order)
# print('Normalized Confusion Matrix: \n',normalized_cm_df,end='\n\n')
#
# ## ---- ----Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm_df,cmap='Greens',fmt='0.2f',annot=True,xticklabels=labels_order,yticklabels=labels_order,cbar_kws={'label':'Color Bar',"orientation":"vertical"})
# plt.title('Normalized Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.tight_layout()
# # plt.show()



## 14 recall, precision, specificity-shortcomings

'''In this video we understood that we should never persist on only one kind of Confusion matrix property, we should chech CM along to all indeices like 
Recall score, Prcision score, specificity score and F1 Score; which F1 Score will be discussed into the next video.'''


## 15 f1 score

'''F1 Score or F1 Measure is Harmonic Mean of Recall score and Precision score.[ i.e.: (2)/((1/Recall)+(1/Precision)) ] and is used to give us a general 
report of recall an precision indices at once. general formula of Harmonic Mean is :   n/Sigma_i_to_n(1/x_i)  '''
''' we can calculate it like below:'''

# from sklearn.metrics import f1_score
# f1=f1_score(y_test,y_pred,labels=labels_order)
# print('F1 Score: \n',f1,end='\n\n')
#
# plt.show()  ## from above section! not ##15 section.




## 16 harmonic mean:
'''this video was a description of Harmonic Mean and its applicable .'''



## 17 ROC (theory):

'''' In this video talked about TP Rate (TPR) [which is our Recall score.] and FPR (i.e.: FP/TN+FP). they lead us to plot ROC diagram ;
which is TPR w.r.t. FPR and both of axes are earned from a cut-off or Threshold that we set during modeling.
prediction is not only <<<y_pred=model.predict(x_test)>>> ;but also <<<y_pred_prob=model.predict_proba(x_test)>>>; where return a (-1,2)-shaped array,
 which its first column implies (if assume as default, our label order is [0,1], i.e.: being negative or being positive) probablity of being negative
 and second column correspond to being positive. if we set a threshold or a cut-off to one of this column,(one column is enoguh for analysing) the mapped
 answer will produce a FPR and a TPR,and predicted y values would be changed. if we change our range of cut-off in a set of desired value,
 it would return us a range of FPR and TPR and we can plot it as ROC diagram.  '''

## 18-ROC (Python)
'''this video represent ROC plotting and computation with 3 method , AUC ( area of under curve ) and etc.'''


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ################################
# ## Reading Data:
# ################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# #########################
# ## Preprocessing:
# #########################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40)
#
# ## Scaling:
# ## -- Standard Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ########################
# ## Building our model:
# ########################
# ## KNN:
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
#
# ########################
# ## Prediction and Evaluation:
# ########################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test) ## it will produce a 2D array which tell us how much it's possible label (w.r.t. how we define label order) be negative or positive.
#
# ## Evaluation:
# ## ---- accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_test,y_pred)
# print(f'The Accuracy is {acc*100}%',end='\n\n')
# ## ---- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# labels_order=[0,1]
# cm=confusion_matrix(y_test,y_pred,labels=labels_order)
# cm_df=pd.DataFrame(cm,index=labels_order,columns=labels_order)
# print('CM DF: \n',cm_df,end='\n\n')
#
#
# ## ---- ---- Confusion Matrix Recall(Sensitivity),Precision and Specificity:
# ## Recall:
# from sklearn.metrics import recall_score
# recall=recall_score(y_test,y_pred,labels=labels_order)
# print('Recall: ',recall,end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_test,y_pred,labels=labels_order)
# print('Precision: ',precision,end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])
# print('Specificity: ',specificity,end='\n\n')
#
# ## F1 Score:
# from sklearn.metrics import f1_score
# f1=f1_score(y_test,y_pred,labels=labels_order)
# print('F1 Score: \n',f1,end='\n\n')
#
# ## ROC  (Reciever Operating Characterictic Curve):
# ##--method (1): <<<<from_predictions>>>
# from sklearn.metrics import RocCurveDisplay
# # plt.figure('ROC_Method_1')
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1]) ## we should insert y_pred_prob[:,1] as y_pred argument. the 1st column ( [:,1] )here represent probablity of being positive.
# plt.title('ROC: Method(1)')
# plt.tight_layout()
#
#
# ##--method (2): <<<from_estimator>>>
# from sklearn.metrics import RocCurveDisplay
# # plt.figure('ROC_Method2')
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test) ## we insert our estimator model (here is KNeighborClassifier(n_neighbors=7)) and our true test data x and y.
# plt.title('ROC: Method(2)')
# plt.tight_layout()
#
# ##--method (3): """the best and recommend way"""
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1]) ## this function return FPR,TPR and Cut-off (threshold)
# plt.figure('ROC_Method3')
# plt.plot(fpr,tpr,c='blueviolet',label='ROC w.r.t. FPR-TPR')
# plt.legend()
# plt.title('ROC: Method(3)')
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.tight_layout()
#
# ##AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print('AUC: \n',auc,end='\n\n')
#
#
#
# ## ---- ---- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm_df,cmap='Greens',fmt='0.2f',annot=True,xticklabels=labels_order,yticklabels=labels_order,cbar_kws={'label':'Color Bar',"orientation":"vertical"})
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.tight_layout()
#
#
# ## ---- Normalized Confusion Matrix:
# normalized_cm=cm/np.sum(cm,axis=1).reshape(-1,1)
# normalized_cm_df=pd.DataFrame(normalized_cm,index=labels_order,columns=labels_order)
# print('Normalized Confusion Matrix: \n',normalized_cm_df,end='\n\n')
#
# ## ---- ----Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm_df,cmap='Greens',fmt='0.2f',annot=True,xticklabels=labels_order,yticklabels=labels_order,cbar_kws={'label':'Color Bar',"orientation":"vertical"})
# plt.title('Normalized Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.tight_layout()
# plt.show()
#
#
#
# ## 19 visualization of the model decision boundaries:
#
# '''this video talk about how to visualize our model (e.g.: KNN) yo shown decision boundaries. it is usful for datasets which their
#  entangled features for learning process are 2 or 3 at last. this give us a 2D or 3D diagram. this video showed that KNN model
#  has a non-linear region and it is not only a single line which divide the area by two. it could be divide area to multiple sectors
#  which gonna correspond being 0 (Negative) or 1 (Positive).'''
#
# ## 20 visualization of the model decision boundaries-complementary
# '''Now let's visualize the decision boundaries of our KNN model which has 2 Feature in this issue, so we can easely visualize it because of being 2D. '''
#
#
# ######################################################
# # vizualization of the model's decision boundries
# ######################################################
# model_name="KNN"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
# plt.scatter(x1.ravel(),x2.ravel())
# plt.show()
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()


## 21 cleaning the codes:
'''as the last video of this chapter, let's clean our code. '''

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ########################
# ## Reading Data:
# ########################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# ###########################
# ## Preprocessing:
# ###########################
#
# ##Train Test Splitting
# from sklearn.model_selection import train_test_split
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=None) ##random state==40 correspond to Koolac.org
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
#
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ###############################
# ## Building The Model:
# ###############################
# ## KNN:
# from sklearn.neighbors import KNeighborsClassifier
#
# model=KNeighborsClassifier(n_neighbors=7) #KNN model with N==7
# model.fit(x_train,y_train)
#
# ################################
# ## Prediction and Evalusation:
# ################################
# ## Prediction:
# y_pred=model.predict(x_test) #predicted target values
# y_pred_prob=model.predict_proba(x_test) # and (x_test.shape[0] x 2) array of probablity of being Zero (Negative) or One (Positive)
#
#
#
# ## Evaluation:
#
# label_order=[0,1]
#
# ## --- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_order)
# cm_df=pd.DataFrame(cm,index=label_order,columns=label_order)
# print('CM DF:\n',cm_df,end='\n\n')
#
# ## --- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_order,columns=label_order)
# print('Normalized CM DF:\n',normalized_cm_df,end='\n\n')
#
#
# ##--- accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ##--- Recall or Sensitivity Score:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Recall : {recall}',end='\n\n')
#
# ##--- Precision Score:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Precision : {precision}',end='\n\n')
#
# ## Specificity Score:
# specificity=cm[0,0]/np.sum(cm[0,:])   # TN/(TN + FP)
# print(f'Specificity : {specificity}',end='\n\n')
#
# ## --- F1 or F-Measure Score:
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'F1 : {f1}',end='\n\n')
#
# ## --- AUC (Area of Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1]) # y_score=y_pred_prob[:,1] as positive or being 1
# print(f'AUC : {auc}',end='\n\n')
#
#
# ## ROC (Receiver Operation Characteristic Curve):
# ## *** method (1) from prediction:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1]) # y_pred=y_pred_prob[:,1] as positive or being 1
# plt.title('ROC (from prediction method)')
# plt.show()
#
# ## *** method (2) from estimator:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC (from estimator method)')
# plt.show()
#
# ## *** method (3) from estimator:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,color='springgreen',label=f"ROC Curve. AUC= {np.round(auc,2)}")
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend()
# plt.grid()
# plt.title('ROC')
# plt.tight_layout()
# plt.show()
#
#
# ## Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# ## Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="KNN"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()




'''P02-03-Logistic Regression'''

## 01 logistic regression:

'''in this singular video of its chapter , talked about logistic regression ( i.e.: 1/(1+exp[-x]) )as another model to fit our data'''

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ########################
# ## Reading Data:
# ########################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# ###########################
# ## Preprocessing:
# ###########################
#
# ##Train Test Splitting
# from sklearn.model_selection import train_test_split
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40) ##random state==40 correspond to Koolac.org
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
#
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ###############################
# ## Building The Model:
# ###############################
# ## Logistic Regression:
# from sklearn.linear_model import LogisticRegression
#
# model=LogisticRegression() #Sigmoid Logistic Regression (1/(1+exp[-x]))
# model.fit(x_train,y_train)
#
# ################################
# ## Prediction and Evaluation:
# ################################
# ## Prediction:
# y_pred=model.predict(x_test) #predicted target values
# y_pred_prob=model.predict_proba(x_test) # and (x_test.shape[0] x 2) array of probablity of being Zero (Negative) or One (Positive)
#
#
#
# ## Evaluation:
#
# label_order=[0,1]
#
# ## --- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_order)
# cm_df=pd.DataFrame(cm,index=label_order,columns=label_order)
# print('CM DF:\n',cm_df,end='\n\n')
#
# ## --- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_order,columns=label_order)
# print('Normalized CM DF:\n',normalized_cm_df,end='\n\n')
#
#
# ##--- accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ##--- Recall or Sensitivity Score:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Recall : {recall}',end='\n\n')
#
# ##--- Precision Score:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Precision : {precision}',end='\n\n')
#
# ## Specificity Score:
# specificity=cm[0,0]/np.sum(cm[0,:])   # TN/(TN + FP)
# print(f'Specificity : {specificity}',end='\n\n')
#
# ## --- F1 or F-Measure Score:
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'F1 : {f1}',end='\n\n')
#
# ## --- AUC (Area of Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1]) # y_score=y_pred_prob[:,1] as positive or being 1
# print(f'AUC : {auc}',end='\n\n')
#
#
# ## ROC (Receiver Operation Characteristic Curve):
# ## *** method (1) from prediction:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1]) # y_pred=y_pred_prob[:,1] as positive or being 1
# plt.title('ROC (from prediction method)')
# plt.show()
#
# ## *** method (2) from estimator:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC (from estimator method)')
# plt.show()
#
# ## *** method (3) from estimator:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,color='springgreen',label=f"ROC Curve. AUC= {np.round(auc,2)}")
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend()
# plt.grid()
# plt.title('ROC')
# plt.tight_layout()
# plt.show()
#
#
# ## Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# ## Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="Logistic Regression"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()



'''P02-04-Decision Tree'''

## 01 Decision Tree-Concept:
'''In this video, has been explained about Decision Tree concept (model).
Decion tree models are not vulnerable to scaling , but it's better to scale our data.'''


## 02 Decision Tree-Python:

'''now let's utilize Decision Tree in our code as model'''




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ########################
# ## Reading Data:
# ########################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# ###########################
# ## Preprocessing:
# ###########################
#
# ##Train Test Splitting
# from sklearn.model_selection import train_test_split
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40) ##random state==40 correspond to Koolac.org
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
#
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ###############################
# ## Building The Model:
# ###############################
# ## Descision Tree::
# from sklearn.tree import DecisionTreeClassifier
#
# model=DecisionTreeClassifier() #Decision Tree model (Note that decion tree models are not vulnerable to scaling , but it's better to scale our data. )
# model.fit(x_train,y_train)
#
# ################################
# ## Prediction and Evalusation:
# ################################
# ## Prediction:
# y_pred=model.predict(x_test) #predicted target values
# y_pred_prob=model.predict_proba(x_test) # and (x_test.shape[0] x 2) array of probablity of being Zero (Negative) or One (Positive)
#
#
#
# ## Evaluation:
#
# label_order=[0,1]
#
# ## --- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_order)
# cm_df=pd.DataFrame(cm,index=label_order,columns=label_order)
# print('CM DF:\n',cm_df,end='\n\n')
#
# ## --- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_order,columns=label_order)
# print('Normalized CM DF:\n',normalized_cm_df,end='\n\n')
#
#
# ##--- accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ##--- Recall or Sensitivity Score:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Recall : {recall}',end='\n\n')
#
# ##--- Precision Score:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Precision : {precision}',end='\n\n')
#
# ## Specificity Score:
# specificity=cm[0,0]/np.sum(cm[0,:])   # TN/(TN + FP)
# print(f'Specificity : {specificity}',end='\n\n')
#
# ## --- F1 or F-Measure Score:
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'F1 : {f1}',end='\n\n')
#
# ## --- AUC (Area of Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1]) # y_score=y_pred_prob[:,1] as positive or being 1
# print(f'AUC : {auc}',end='\n\n')
#
#
# ## ROC (Receiver Operation Characteristic Curve):
# ## *** method (1) from prediction:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1]) # y_pred=y_pred_prob[:,1] as positive or being 1
# plt.title('ROC (from prediction method)')
# plt.show()
#
# ## *** method (2) from estimator:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC (from estimator method)')
# plt.show()
#
# ## *** method (3) from estimator:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,color='springgreen',label=f"ROC Curve. AUC= {np.round(auc,2)}")
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend()
# plt.grid()
# plt.title('ROC')
# plt.tight_layout()
# plt.show()
#
#
# ## Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# ## Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="Decision Tree"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()



## 03 Decision Tree-plot:


'''let's plot descision tree among our code.'''



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# ########################
# ## Reading Data:
# ########################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1] #input
# y=df.iloc[:,-1] #output
#
# ###########################
# ## Preprocessing:
# ###########################
#
# ##Train Test Splitting
# from sklearn.model_selection import train_test_split
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40) ##random state==40 correspond to Koolac.org
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
#
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
# ###############################
# ## Building The Model:
# ###############################
# ## Descision Tree::
# from sklearn.tree import DecisionTreeClassifier ,plot_tree #we also add plot tree to have better visalizataion of our decision tree algorithm
#
# model=DecisionTreeClassifier(max_depth=None) #Decision Tree model (Note that decion tree models are not vulnerable to scaling , but it's better to scale our data. )
# model.fit(x_train,y_train)
# plt.figure('Decision Tree',figsize=[14,5])
# plot_tree(decision_tree=model,max_depth=None,feature_names=['Age','Estimated Salary'],class_names=['None-Buyer','Buyer'],fontsize=5,filled=True) # we can change our max depth,feature name
# # ( i.e.: here are Age and Estimated Salary) ,predicted or the final decision of decision tree algorithm by <<lass name>> as our target value
# # (i.e.: here are None-Buyer {as 0}, Buyer {as 1}), colorify our chart by <<filled>>
# plt.tight_layout()
# plt.show()
#
#
# ################################
# ## Prediction and Evalusation:
# ################################
# ## Prediction:
# y_pred=model.predict(x_test) #predicted target values
# y_pred_prob=model.predict_proba(x_test) # and (x_test.shape[0] x 2) array of probablity of being Zero (Negative) or One (Positive)
#
#
#
# ## Evaluation:
#
# label_order=[0,1]
#
# ## --- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_order)
# cm_df=pd.DataFrame(cm,index=label_order,columns=label_order)
# print('CM DF:\n',cm_df,end='\n\n')
#
# ## --- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_order,columns=label_order)
# print('Normalized CM DF:\n',normalized_cm_df,end='\n\n')
#
#
# ##--- accuracy:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ##--- Recall or Sensitivity Score:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Recall : {recall}',end='\n\n')
#
# ##--- Precision Score:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'Precision : {precision}',end='\n\n')
#
# ## Specificity Score:
# specificity=cm[0,0]/np.sum(cm[0,:])   # TN/(TN + FP)
# print(f'Specificity : {specificity}',end='\n\n')
#
# ## --- F1 or F-Measure Score:
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_order)
# print(f'F1 : {f1}',end='\n\n')
#
# ## --- AUC (Area of Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1]) # y_score=y_pred_prob[:,1] as positive or being 1
# print(f'AUC : {auc}',end='\n\n')
#
#
# ## ROC (Receiver Operation Characteristic Curve):
# ## *** method (1) from prediction:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1]) # y_pred=y_pred_prob[:,1] as positive or being 1
# plt.title('ROC (from prediction method)')
# plt.show()
#
# ## *** method (2) from estimator:
# from sklearn.metrics import RocCurveDisplay
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC (from estimator method)')
# plt.show()
#
# ## *** method (3) from estimator:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,color='springgreen',label=f"ROC Curve. AUC= {np.round(auc,2)}")
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend()
# plt.grid()
# plt.title('ROC')
# plt.tight_layout()
# plt.show()
#
#
# ## Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# ## Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',xticklabels=label_order,yticklabels=label_order,cbar_kws={"orientation":"vertical",'label':'Color Bar'})
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.tight_layout()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="Decision Tree"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()


## 04 Decision Tree-plot (complementary):
'''this video was about another approach to plot our decision tree.
we use <<<from sklearn.tree import export_graphviz>>> , then we open a new text file with "w" mode and complete the properties of our desired chart.
e.g.: with open("H:/Koolac/DT/tree.txt","w") as f:
    export_graphviz(model,out_file=f,feature_names=["Age","Estimated Salary"],filled=True,class_names=["non-buyer","buyer"])'''




'''P02-05-Random Forest'''


## 01 Random Forest-theory:

'''in this video random forest algorthm has been explained. first, we choose number of estimator (i.e.: number of all trees ), secondly we randomly 
pick up data "K" times from our training set "with INPLACEMENT" and with this new table we build a decision tree,
third, we repeat step 2 n times (i.e.: numbers of all trees). finaly the decision is made by mostly commons trees'es decison.   
 '''


## 02 Random Forest-Python:
'''let's write the python code of above description as an example.'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Random Forest Model:
# from sklearn.ensemble import RandomForestClassifier
# model=RandomForestClassifier(n_estimators=100,max_depth=3) # in next videos we will notice that sometimes if we choose maximum depth properly,
# # algorithm will drive us to better condition with less error.
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="Random Forest"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()




"""P02-06-Overfit and Underfit + Random State"""

## 01 Overfit and Underfit:

'''overfitting , underfitting and good fitting concepts is explained in this video'''


## 02 random_state:

'''Some algorithms like KNN doesn't have random processes in itself, so random_state just can be applied in our train-test splitting preprocessing.
 but others like 'DecisionTree', and of course , "Random Forest", and even "Logistic Regression" have stochastic processes in itself.
  we can set a random_state for this model but it is not recommended, we can run our code in a loop and take an average from its F1 score or Accuracy score
   or others approach which will be explained further.'''








'''P02-07-Naive Bayes'''

## 01 probability review-probability concept:

## 02 probability review-conditional probability:

## 03 probability review-bayes theorem:

## 04 probability review-law of total probability:

## 05 probability review-independence:

'''In these videos intros of probability , Bayesian probability fundamentals and law of total probability have been discussed.
In next videos Naive Bayes algorithm will be explained.'''

## 06 Naive Bayes:

## 07 Naive Bayes-Complementary:

'''Naive means weak, or loose and.... . if you think that you need a recall or remembrance , watch these two vidos again.'''


## 08 Naive Bayes-Python:

'''Now let's code a Naive Bayes model algorithms '''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Naive-Bayes Model:
# from mixed_naive_bayes import MixedNB
# model=MixedNB(categorical_features=None,alpha=1) # generarly , we insert the catagorical features as list ; it's correspond to their mark of columns.
# # in this problem, there are NO "Categorical" features (i.e.: there are only "Numeric" feature),
# # so we don't need this option to mention or we can set it to "None". If all of features are Categorical, then we set <<<categorical_features = 'all'>>>.
# # also we apply alpha==1 (i.e.: Laplace smoothing). if we set alpha < 1 , we call it Lidstone Smoothing. the default value of alpha in this module
# # has set to 0.5.
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# # ## --- ROC with estimation method:     (    DOESN'T WORK WITH MIXED-NAIVE-BAYES MODULE    )
# # RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# # plt.title('ROC Curve (From Estimation Method)')
# # plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="Naive-Bayes"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()





'''P02-08-SVM   (Support Vector Machine)   '''

## 01 SVM:

'''Support Vector Machine (SVM), Maximum Margin Classifier with a Hyperplane ,
Kernel Concept for SVM and Kernel Tricks and some Kernel Function with their "Landmark (i.e.: mean)" and variance like RBF(Gaussian) {Radio Base Function},
and others attributes like Sigmoid (tanh form) , and Polynomial kernel has been explained.'''


## 02-SVM-Python:
'''let's code !'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Support Vector Machine (SVM):
# from sklearn.svm import SVC # i.e.: Support Vector Classifier
# model=SVC(kernel='rbf',degree=0,probability=True) ## For example, we can select our kernel from:
# ## <<<kernel="rbf">>> (i.e: Radio Base Function [this is Gaussian])
# ## <<<kernel="linear">>> (i.e: linear model)
# ## <<<kernel="poly">>> (i.e: polynomial model ). In this case also we must set <<<degree =  >>>
#
# ## If we want a predict_prob in Prediction section, we should set <<<probability=True>>>
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="SVM"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()



## 03 SVM-Soft Margin:

'''this video talked about Soft-Margin concept in SVM.
the default value of C=1 (constant for penalty of datas which are in wrong side of hyperplane; when we set a soft margin [sklearn do it as default] instead 
of a regular margin hyperplane, some of datas may stand on wrong side of hyperplane, which we commit them to penalty to change their place.
we do it with C*(sum(xi_i)+sum(xi_j)+...)  [xi is a greek letter]), and should be in (0,inf) interval.'''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Support Vector Machine (SVM):
# from sklearn.svm import SVC # i.e.: Support Vector Classifier
# model=SVC(kernel='rbf',C=10,probability=True) ## For example, we can select our kernel from:
# ## <<<kernel="rbf">>> (i.e: Radio Base Function [this is Gaussian])
# ## <<<kernel="linear">>> (i.e: linear model)
# ## <<<kernel="poly">>> (i.e: polynomial model ). In this case also we must set <<<degree =  >>>
#
# ## If we want a predict_prob in Prediction section, we should set <<<probability=True>>>
#
# ## <<<C = >>> is for penalty which discussed in intro and begining of the code.
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="SVM"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()


'''P02-09-Handling Categorical Data'''

## 01 Nominal-with 2 values:

'''to this far , we only consider two features as input data (i.e.: Age , Estimated Salary). But if we are going to consider others categorical
 (or Nominal) features which described with "non-numeric" values, we should first convert these categorical values to Nominal values.'''

'''For Instance , consider the buyer_non-buyer dataset again:'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Method 1: #### with Pandas module:
#
# transformed_Gender=df.loc[:,'Gender']
# print('Encoded Gender (Before mapping):\n',transformed_Gender,end='\n\n')
# transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
# print('Encoded Gender (After mapping):\n',transformed_Gender,end='\n\n')
# transformed_Gender=transformed_Gender.values.reshape(-1,1) # turn it to a numpy array and make it as (-1,1) array
# print('Encoded Gender (NumPy form):\n',transformed_Gender,end='\n\n')
#
#
# ## Method 2: #### with scikit-learn module:
#
# from sklearn.preprocessing import OrdinalEncoder
# encoder_Gender=OrdinalEncoder()
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# print('Encoded Gender (Scikit_learn form):\n',transformed_Gender,end='\n\n')
# df_new=pd.DataFrame(transformed_Gender,columns=encoder_Gender.get_feature_names_out()) ## if we'd want to make it a DF again ith better visualization
# print('DF of Transformed Gender:\n',df_new,end='\n\n')
# print('What were the actual values of 0 and 1?: \n',encoder_Gender.categories_,end='\n\n') ## using .categoiries_ method
# ## or:
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,['Gender']])
# print('Encoded Gender (Scikit_learn form):\n',transformed_Gender,end='\n\n')
# df_new=pd.DataFrame(transformed_Gender,columns=encoder_Gender.get_feature_names_out()) ## if we'd want to make it a DF again ith better visualization
# print('DF of Transformed Gender:\n',df_new,end='\n\n')
# print('What were the actual values of 0 and 1?: \n',encoder_Gender.categories_,end='\n\n') ## using .categoiries_ method






## 02 Nominal-with more than 2 values:

'''Now what if the categorical columns has more than two nominal values? here , we must utilize "One Hot Encoder" method matrix , then drop one of its columns,
because if we don't drop it , the algorithm may not work properly. so we drop one of its columns. it can be first, last or any of the columns. we hear 
  this dropping as dummy variable. Some people call One Hot encoder as dummy too!  '''
'''See the Example:'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Method 1: #### with Pandas module:
#
# transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True) ## add <<<prefix = >>> to return the dummies matrix with desired prefix before the attributes.
# ## here, the "Type" is name of that categorical feature columns with more than 2 nominal which we wanna transfer it to digit variables
# ## we drop the first columns to turn our "hot one" matrix onto a dummies one.
# print('Transformed Type: (Pandas Method)\n',transformed_Type,end='\n\n')
#
#
# ## Method 2: #### with scikit-learn module:
# from sklearn.preprocessing import OneHotEncoder
# encoder_Type=OneHotEncoder(drop='first',sparse=False) ## we first must change sparse to False (i.e.:<<<sparse=False>>>) if we don't do this , it returns a complicated matrix
# # instead of One Hot Encoder. in Pandas method, the default variable of sparse is False. We drop the first column.
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1)) ## don't forget to convert it to a numpy array then (-1,1) shape!
# print('Transformed Type: (scikit-learn method)\n',transformed_Type,end='\n\n')
# df_new=pd.DataFrame(transformed_Type,columns=encoder_Type.get_feature_names_out()) ## if we'd want to make it a pandas DF again ith better visualization
# print('DF of Transformed Type:\n',df_new,end='\n\n')
# transformed_Type=encoder_Type.fit_transform(df.loc[:,['Type']]) ## Or we can write it like this instead of above line.
# print('Transformed Type: (scikit-learn method)\n',transformed_Type,end='\n\n')
# df_new=pd.DataFrame(transformed_Type,columns=encoder_Type.get_feature_names_out()) ## if we'd want to make it a pandas DF again ith better visualization
# print('DF of Transformed Type:\n',df_new,end='\n\n')




## 03 Ordinal:

'''Ordinal and Nominal are both from categorical group , but in Nominal features the order is not defined and sensible , so we use somthing
like One Hot Encoder matrix. but if our feature the order is describable, we use OrdinalEncoder , just like binary nominal.'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\02-SimpleChurn.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# #############################
# ## Preprocessing:
# ##############################
#
# ## Method 1: #### with Pandas module:
#
# d={
#     'Very Happy':0,
#     'Happy':1,
#     'OK':2,
#     'Unhappy':3,
#     'Very Unhappy':4
# }
#
# transformed_Satisfaction=df.loc[:,'Satisfaction'].map(d)
# print('Transformed Satisfaction: (Pandas Method)\n',transformed_Satisfaction,end='\n\n')
#
#
# ## Method 2: #### with scikit-learn module:
#
# from sklearn.preprocessing import OrdinalEncoder
# encoder_Satisfaction=OrdinalEncoder()
# transformed_Satisfaction=encoder_Satisfaction.fit_transform(df.loc[:,'Satisfaction'].values.reshape(-1, 1)) ## don't forget to convert it to a numpy array then (-1,1) shape!
# print('Transformed Type: (scikit-learn method)\n',transformed_Satisfaction,end='\n\n')
# print('What were the actual name of the encoded ordinal correspondingly? \n\n',encoder_Satisfaction.categories_,end='\n\n')



## 04 considering categorical features-A:

'''Now let's apply it in our Ad.csv dataset.'''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
#
# ## Defining x and y:
# y=df.iloc[:,-1].values #output (target value)
#
#
# ## ---- encoding the 'Gender' column:  (with scikit-learn method)
# encoder_Gender=OrdinalEncoder()  ## Note that it's a nominal with 2 values, so we can work on it with Ordinal's method.
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# # ## ---- encoding the 'Gender' column:  (with Pandas method)
# # transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
#
#
#
# ## ---- encoding the 'Type' column:  (with scikit-learn method)
# encoder_Type=OneHotEncoder(drop='first',sparse_output=False)
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1))
# # ## ---- encoding the 'Type' column:  (with Pandas method)
# # transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True)
#
# ## ---- numeric features:
# numeric_features=df.iloc[:,3:-1].values
#
# ## ---- concatenation: (with NumPy method)
# x=np.concatenate([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # ## ---- concatenation: (with Pandas method)
# # df_new=pd.concat([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # x=df_new.values
#
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# '''!!!!! -- NOTE -- !!!!!'''
# ## we should check whether or "Categorical" columns need scaling or not depend on the problem. or what scaler (i.e.: MinMaxScaler, StandardScaler,..) is
# ## required.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Support Vector Machine (SVM):
# from sklearn.svm import SVC # i.e.: Support Vector Classifier
# model=SVC(kernel='rbf',C=10,probability=True) ## For example, we can select our kernel from:
# ## <<<kernel="rbf">>> (i.e: Radio Base Function [this is Gaussian])
# ## <<<kernel="linear">>> (i.e: linear model)
# ## <<<kernel="poly">>> (i.e: polynomial model ). In this case also we must set <<<degree =  >>>
#
# ## If we want a predict_prob in Prediction section, we should set <<<probability=True>>>
#
# ## <<<C = >>> is for penalty which discussed in intro and begining of the code.
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()







## 05 considering categorical features-B:

'''In this video we learnt that we should check whether scale our categorical columns or not. in last video, we saw that some score metrics are not suitable
 and good for SVM model. now we wanna check that if we don't scale our categorical columns, is it affect our model toward better performance or not.
 and we see that in THIS model is true.
 we also perform this checking in others algorithms like Random Forest , Decision Tree , Logistic Regression , KNN and ... and Naive Bayes.
 There is remark and note in Naive Bayes Algorthm that we should notice:
 1: 
        In Naive Bayes model we must NOT use OneHotEncoder matrix method. NEVER!! we ONLY are authorized to use OrdinalEncoder because we can't give 
 a dataset to a model with stringly form.
 2: 
        We shouldn't scale our categorical model in mixed-naive-bayes module since in this package inserted categorical must be in [0,inf) interval
 range.'''


## For SVM:

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
#
# ## Defining x and y:
# y=df.iloc[:,-1].values #output (target value)
#
#
# ## ---- encoding the 'Gender' column:  (with scikit-learn method)
# encoder_Gender=OrdinalEncoder()  ## Note that it's a nominal with 2 values, so we can work on it with Ordinal's method.
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# # ## ---- encoding the 'Gender' column:  (with Pandas method)
# # transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
#
#
#
# ## ---- encoding the 'Type' column:  (with scikit-learn method)
# encoder_Type=OneHotEncoder(drop='first',sparse_output=False)
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1))
# # ## ---- encoding the 'Type' column:  (with Pandas method)
# # transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True)
#
# ## ---- numeric features:
# numeric_features=df.iloc[:,3:-1].values.reshape(-1,2)
#
# ## ---- concatenation: (with NumPy method)
# x=np.concatenate([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # ## ---- concatenation: (with Pandas method)
# # df_new=pd.concat([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # x=df_new.values
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
#
# ## Scaling:
# '''!!!!! -- NOTE -- !!!!!'''
# ## we should check whether or "Categorical" columns need scaling or not depend on the problem. or what scaler (i.e.: MinMaxScaler, StandardScaler,..) is
# ## required.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train[:,-2:]=scaler.fit_transform(x_train[:,-2:])
# x_test[:,-2:]=scaler.transform(x_test[:,-2:])
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Support Vector Machine (SVM):
# from sklearn.svm import SVC # i.e.: Support Vector Classifier
# model=SVC(kernel='rbf',C=10,probability=True) ## For example, we can select our kernel from:
# ## <<<kernel="rbf">>> (i.e: Radio Base Function [this is Gaussian])
# ## <<<kernel="linear">>> (i.e: linear model)
# ## <<<kernel="poly">>> (i.e: polynomial model ). In this case also we must set <<<degree =  >>>
#
# ## If we want a predict_prob in Prediction section, we should set <<<probability=True>>>
#
# ## <<<C = >>> is for penalty which discussed in intro and begining of the code.
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
#
# ## For KNN k=7
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
#
# ## Defining x and y:
# y=df.iloc[:,-1].values #output (target value)
#
#
# ## ---- encoding the 'Gender' column:  (with scikit-learn method)
# encoder_Gender=OrdinalEncoder()  ## Note that it's a nominal with 2 values, so we can work on it with Ordinal's method.
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# # ## ---- encoding the 'Gender' column:  (with Pandas method)
# # transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
#
#
#
# ## ---- encoding the 'Type' column:  (with scikit-learn method)
# encoder_Type=OneHotEncoder(drop='first',sparse_output=False)
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1))
# # ## ---- encoding the 'Type' column:  (with Pandas method)
# # transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True)
#
# ## ---- numeric features:
# numeric_features=df.iloc[:,3:-1].values
#
# ## ---- concatenation: (with NumPy method)
# x=np.concatenate([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # ## ---- concatenation: (with Pandas method)
# # df_new=pd.concat([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # x=df_new.values
#
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# '''!!!!! -- NOTE -- !!!!!'''
# ## we should check whether or "Categorical" columns need scaling or not depend on the problem. or what scaler (i.e.: MinMaxScaler, StandardScaler,..) is
# ## required.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train[:,-2:]=scaler.fit_transform(x_train[:,-2:])
# x_test[:,-2:]=scaler.transform(x_test[:,-2:])
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## KNN:
# from sklearn.neighbors import KNeighborsClassifier
# model=KNeighborsClassifier(n_neighbors=7)
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
#
#
# ## Random Forest:
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
#
# ## Defining x and y:
# y=df.iloc[:,-1].values #output (target value)
#
#
# ## ---- encoding the 'Gender' column:  (with scikit-learn method)
# encoder_Gender=OrdinalEncoder()  ## Note that it's a nominal with 2 values, so we can work on it with Ordinal's method.
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# # ## ---- encoding the 'Gender' column:  (with Pandas method)
# # transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
#
#
#
# ## ---- encoding the 'Type' column:  (with scikit-learn method)
# encoder_Type=OneHotEncoder(drop='first',sparse_output=False)
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1))
# # ## ---- encoding the 'Type' column:  (with Pandas method)
# # transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True)
#
# ## ---- numeric features:
# numeric_features=df.iloc[:,3:-1].values
#
# ## ---- concatenation: (with NumPy method)
# x=np.concatenate([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # ## ---- concatenation: (with Pandas method)
# # df_new=pd.concat([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # x=df_new.values
#
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# '''!!!!! -- NOTE -- !!!!!'''
# ## we should check whether or "Categorical" columns need scaling or not depend on the problem. or what scaler (i.e.: MinMaxScaler, StandardScaler,..) is
# ## required.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train[:,-2:]=scaler.fit_transform(x_train[:,-2:])
# x_test[:,-2:]=scaler.transform(x_test[:,-2:])
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Random Forest:
# from sklearn.ensemble import RandomForestClassifier
# model=RandomForestClassifier(n_estimators=100,max_depth=3)
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
#
#
#
#
# ## Decision Tree:
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
#
# ## Defining x and y:
# y=df.iloc[:,-1].values #output (target value)
#
#
# ## ---- encoding the 'Gender' column:  (with scikit-learn method)
# encoder_Gender=OrdinalEncoder()  ## Note that it's a nominal with 2 values, so we can work on it with Ordinal's method.
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# # ## ---- encoding the 'Gender' column:  (with Pandas method)
# # transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
#
#
#
# ## ---- encoding the 'Type' column:  (with scikit-learn method)
# encoder_Type=OneHotEncoder(drop='first',sparse_output=False)
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1))
# # ## ---- encoding the 'Type' column:  (with Pandas method)
# # transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True)
#
# ## ---- numeric features:
# numeric_features=df.iloc[:,3:-1].values
#
# ## ---- concatenation: (with NumPy method)
# x=np.concatenate([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # ## ---- concatenation: (with Pandas method)
# # df_new=pd.concat([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # x=df_new.values
#
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# '''!!!!! -- NOTE -- !!!!!'''
# ## we should check whether or "Categorical" columns need scaling or not depend on the problem. or what scaler (i.e.: MinMaxScaler, StandardScaler,..) is
# ## required.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train[:,-2:]=scaler.fit_transform(x_train[:,-2:])
# x_test[:,-2:]=scaler.transform(x_test[:,-2:])
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Decision Tree:
# from sklearn.tree import DecisionTreeClassifier,plot_tree
# model=DecisionTreeClassifier(max_depth=3)
# model.fit(x_train,y_train)
# plot_tree(decision_tree=model,feature_names=None,class_names=['None-Buyer','Buyer'],filled=True)
# plt.show()
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
#
#
#
# ## Logistic Regression:
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
#
# ## Defining x and y:
# y=df.iloc[:,-1].values #output (target value)
#
#
# ## ---- encoding the 'Gender' column:  (with scikit-learn method)
# encoder_Gender=OrdinalEncoder()  ## Note that it's a nominal with 2 values, so we can work on it with Ordinal's method.
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# # ## ---- encoding the 'Gender' column:  (with Pandas method)
# # transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
#
#
#
# ## ---- encoding the 'Type' column:  (with scikit-learn method)
# encoder_Type=OneHotEncoder(drop='first',sparse_output=False)
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1))
# # ## ---- encoding the 'Type' column:  (with Pandas method)
# # transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True)
#
# ## ---- numeric features:
# numeric_features=df.iloc[:,3:-1].values
#
# ## ---- concatenation: (with NumPy method)
# x=np.concatenate([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # ## ---- concatenation: (with Pandas method)
# # df_new=pd.concat([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # x=df_new.values
#
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# '''!!!!! -- NOTE -- !!!!!'''
# ## we should check whether or "Categorical" columns need scaling or not depend on the problem. or what scaler (i.e.: MinMaxScaler, StandardScaler,..) is
# ## required.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train[:,-2:]=scaler.fit_transform(x_train[:,-2:])
# x_test[:,-2:]=scaler.transform(x_test[:,-2:])
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Logistic Regression:
# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression()
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
#
#
# ## Naive Bayes:
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# from sklearn.preprocessing import OrdinalEncoder
#
# ## Defining x and y:
# y=df.iloc[:,-1].values #output (target value)
#
#
# ## ---- encoding the 'Gender' column:  (with scikit-learn method)
# encoder_Gender=OrdinalEncoder()
# transformed_Gender=encoder_Gender.fit_transform(df.loc[:,'Gender'].values.reshape(-1,1))
# # ## ---- encoding the 'Gender' column:  (with Pandas method)
# # transformed_Gender=df.loc[:,'Gender'].map({'Female':0,'Male':1})
#
#
#
# ## ---- encoding the 'Type' column:  (with scikit-learn method)
# encoder_Type=OrdinalEncoder()  ## it's naive bayes. nominal columns should be treated as ordinal columns!!
# transformed_Type=encoder_Type.fit_transform(df.loc[:,'Type'].values.reshape(-1,1))
# # ## ---- encoding the 'Type' column:  (with Pandas method)
# # transformed_Type=pd.get_dummies(df.loc[:,'Type'],prefix='Type',drop_first=True)
#
# ## ---- numeric features:
# numeric_features=df.iloc[:,3:-1].values
#
# ## ---- concatenation: (with NumPy method)
# x=np.concatenate([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # ## ---- concatenation: (with Pandas method)
# # df_new=pd.concat([transformed_Gender,transformed_Type,numeric_features],axis=1)
# # x=df_new.values
#
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# '''!!!!! -- NOTE -- !!!!!'''
# ## we should check whether or "Categorical" columns need scaling or not depend on the problem. or what scaler (i.e.: MinMaxScaler, StandardScaler,..) is
# ## required.
# ## Categorical columns in Naive Bayes model shouldn't be scaled
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train[:,-2:]=scaler.fit_transform(x_train[:,-2:])
# x_test[:,-2:]=scaler.transform(x_test[:,-2:])
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Naive Bayes
# from mixed_naive_bayes import MixedNB
# model=MixedNB(categorical_features=[0,1],alpha=1) ## our categorical features are columns 0 and 1. alpha==1 is Laplace smoothing
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
# # ## --- ROC with estimation method:                ## Doesn't work with mixed_naive_bayes module.
# # RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# # plt.title('ROC Curve (From Estimation Method)')
# # plt.show()
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()








'''P02-10-Cross Validation'''

## 01 Cross Validation-Concept:

'''We know that in train-test-splitting preprocessing section there is a random_state argument where can be set on a specific number to avoid stochastic 
 selection process in each run. but some algorthms have stochastic and random processes in their creation; like Random Forest and ... . some doesn't,
 like KNN. In this lecture we are going to seek those methods which give us a criteria to express our model, Good? Bad? and etc. to avoid these 
 random steps in each code run, we need a deterministic criteria , which is discussed in "Cross Validation" concept.'''



## 02 Cross Validation-theory:

'''K-Fold Cross Validation has been explained in this video. we split our dataset to K part. (usually is splitted to 5, 3, or 10 depends on size of
dataset.) then 1/K of dataset goes to test data and rest goes ro train. we calculate one of the score metrics (e.g.: F1 Score) for our test data 
and call it score_1. then we (K-1) other time repeat this whole process till we calculate score_K. Finally , we take an average (i.e.: Mean)
onto our scores that we computed. and now we can have a better validation on our model, which call it "K-Fold Cross Validation".
 Note that there are more methods like K-Fold but they're not essential to study in this course.'''



## 03 Cross Validation-Python:

'''Now let's code !'''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##################################
# ## Cross Validation:
# ##################################
#
#
# from sklearn.model_selection import KFold ## we use KFold method for cross validating our model
# cv=KFold(n_splits=5,shuffle=True) ## we MUST shuffle our input data , we discussed it in the earliest videos
#
# ## we make some empty list class which is useful in end of the code :
#
# acc_avg=[]
# recall_avg=[]
# precision_avg=[]
# specificity_avg=[]
# f1_avg=[]
# auc_avg=[]
#
# for train_idx,test_idx in cv.split(x): ## now we split our input data
#     ##############################
#     ## Preprocessing:
#     ##############################
#
#     x_train,x_test = x[train_idx,:],x[test_idx,:] ## all columns correspond to train and test row's specific index are our x_train and x_test.
#     y_train,y_test = y[train_idx],y[test_idx] ## ""  ""  "" "". Note that "y" is a 1D array, so it doesn't need a coulumn index.
#
#
#     ## Scaling:
#     from sklearn.preprocessing import StandardScaler
#     scaler=StandardScaler()
#     x_train=scaler.fit_transform(x_train)
#     x_test=scaler.transform(x_test)
#
#
#     ##############################
#     ## Building Model:
#     ##############################
#
#
#     from sklearn.ensemble import RandomForestClassifier
#     model=RandomForestClassifier(n_estimators=100,max_depth=3)
#     model.fit(x_train,y_train)
#
#
#     ##############################
#     ## Prediction and Evaluation:
#     ##############################
#
#     ## Prediction:
#     y_pred=model.predict(x_test)
#     y_pred_prob=model.predict_proba(x_test)
#
#     ## Evaluation:
#     label_orders=[0,1]
#
#     ## -- Confusion Matrix:
#     from sklearn.metrics import confusion_matrix
#     cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
#     cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
#     print('Confusion Matrix:\n',cm_df,end='\n\n')
#
#     ## -- Normalized Confusion Matrix:
#     normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
#     normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
#     print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
#     # ## -- -- Confusion Matrix Heatmap:
#     # import seaborn as sns
#     # plt.figure('Confusion Matrix')
#     # sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
#     # plt.xlabel('Predicted')
#     # plt.ylabel('Actual')
#     # plt.title('Confusion Matrix')
#     # plt.show()
#
#
#     # ## -- -- Normalized Confusion Matrix Heatmap:
#     # import seaborn as sns
#     # plt.figure('Normalized Confusion Matrix')
#     # sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
#     # plt.xlabel('Predicted')
#     # plt.ylabel('Actual')
#     # plt.title('Normalized Confusion Matrix')
#     # plt.show()
#
#     ## Accuracy Score:
#     from sklearn.metrics import accuracy_score
#     acc=accuracy_score(y_true=y_test,y_pred=y_pred)
#     acc_avg.append(acc)
#     print(f'Accuracy: {acc}',end='\n\n')
#
#     ## Recall or Sensitivity:
#     from sklearn.metrics import recall_score
#     recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
#     recall_avg.append(recall)
#     print(f'Recall: {recall}',end='\n\n')
#
#     ## Precision:
#     from sklearn.metrics import precision_score
#     precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
#     precision_avg.append(precision)
#     print(f'Precision: {precision}',end='\n\n')
#
#     ## Specificity:
#     specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
#     specificity_avg.append(specificity)
#     print(f'Specificity: {specificity}',end='\n\n')
#
#     ## F1 Score (F-Measure):
#     from sklearn.metrics import f1_score
#     f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
#     f1_avg.append(f1)
#     print(f'F1: {f1}',end='\n\n')
#
#     ## AUC (Area Under Curve):
#     from sklearn.metrics import roc_auc_score
#     auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
#     auc_avg.append(auc)
#     print(f'AUC: {auc}',end='\n\n')
#
#     ## ROC (Receiver Operation Characteristic Curve):
#     # from sklearn.metrics import RocCurveDisplay
#     #
#     # ## --- ROC with prediction method:
#     # RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
#     # plt.title('ROC Curve (From Prediction Method)')
#     # plt.show()
#     # ## --- ROC with estimation method:
#     # RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
#     # plt.title('ROC Curve (From Estimation Method)')
#     # plt.show()
#     # ## --- ROC with General Method:
#     # from sklearn.metrics import roc_curve
#     # fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
#     # plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
#     # plt.legend()
#     # plt.xlabel('False Positive Rate')
#     # plt.ylabel('True Positive Rate')
#     # plt.title('ROC')
#     # plt.grid()
#     # plt.show()
#     #
#     #
#     # #######################################
#     # ## Visualization of model's decision boundaries:
#     # #######################################
#     #
#     # model_name="SVM"
#     # is_scaled=True
#     #
#     # # --- train
#     # from matplotlib.colors import ListedColormap
#     # cmap=ListedColormap(["red","green"])
#     #
#     # x_set,y_set=x_train,y_train
#     #
#     # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#     #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#     #
#     # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
#     # plt.xlim(x1.min(),x1.max())
#     # plt.ylim(x2.min(),x2.max())
#     # for i, j in enumerate(np.unique(y_set)):
#     #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
#     # plt.title(f"{model_name} (Training set)")
#     # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
#     # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
#     # plt.legend()
#     # plt.show()
#     #
#     # # --- test
#     # from matplotlib.colors import ListedColormap
#     # cmap=ListedColormap(["red","green"])
#     #
#     # x_set,y_set=x_test,y_test
#     #
#     # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#     #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#     #
#     # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
#     # plt.xlim(x1.min(),x1.max())
#     # plt.ylim(x2.min(),x2.max())
#     # for i, j in enumerate(np.unique(y_set)):
#     #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
#     # plt.title(f"{model_name} (Test set)")
#     # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
#     # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
#     # plt.legend()
#     # plt.show()
# print('====================================================================',end='\n\n')
# print(f'Accuracy Mean: {np.mean(acc_avg)}')
# print(f'Recall Mean: {np.mean(recall_avg)}')
# print(f'Precision Mean: {np.mean(precision_avg)}')
# print(f'Specificity Mean: {np.mean(specificity_avg)}')
# print(f'F1 Mean: {np.mean(f1_avg)}')
# print(f'AUC Mean: {np.mean(auc_avg)}')




## 04 Cross Validation-Stratified KFold:

'''This video is about Stratified K-Fold cross valdiation. if we're up to comply x to y ratio while selecting train data, instead KFold ,we use 
"StratifiedKFold".  '''



# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##################################
# ## Cross Validation:
# ##################################
#
#
# from sklearn.model_selection import StratifiedKFold ## we use KFold method for cross validating our model
# cv=StratifiedKFold(n_splits=5,shuffle=True) ## we MUST shuffle our input data , we discussed it in the earliest videos
#
# ## we make some empty list class which is useful in end of the code :
#
# acc_avg=[]
# recall_avg=[]
# precision_avg=[]
# specificity_avg=[]
# f1_avg=[]
# auc_avg=[]
#
# for train_idx,test_idx in cv.split(x,y): ## now we split our input data.           !!! -- NOTE THAT WE ADD "y" IN StratifiedKFold SPLITTER TOO -- !!!
#     ##############################
#     ## Preprocessing:
#     ##############################
#
#     x_train,x_test = x[train_idx,:],x[test_idx,:] ## all columns correspond to train and test row's specific index are our x_train and x_test.
#     y_train,y_test = y[train_idx],y[test_idx] ## ""  ""  "" "". Note that "y" is a 1D array, so it doesn't need a coulumn index.
#
#
#     ## Scaling:
#     from sklearn.preprocessing import StandardScaler
#     scaler=StandardScaler()
#     x_train=scaler.fit_transform(x_train)
#     x_test=scaler.transform(x_test)
#
#
#     ##############################
#     ## Building Model:
#     ##############################
#
#
#     from sklearn.ensemble import RandomForestClassifier
#     model=RandomForestClassifier(n_estimators=100,max_depth=3)
#     model.fit(x_train,y_train)
#
#
#     ##############################
#     ## Prediction and Evaluation:
#     ##############################
#
#     ## Prediction:
#     y_pred=model.predict(x_test)
#     y_pred_prob=model.predict_proba(x_test)
#
#     ## Evaluation:
#     label_orders=[0,1]
#
#     ## -- Confusion Matrix:
#     from sklearn.metrics import confusion_matrix
#     cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
#     cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
#     print('Confusion Matrix:\n',cm_df,end='\n\n')
#
#     ## -- Normalized Confusion Matrix:
#     normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
#     normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
#     print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
#     # ## -- -- Confusion Matrix Heatmap:
#     # import seaborn as sns
#     # plt.figure('Confusion Matrix')
#     # sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
#     # plt.xlabel('Predicted')
#     # plt.ylabel('Actual')
#     # plt.title('Confusion Matrix')
#     # plt.show()
#
#
#     # ## -- -- Normalized Confusion Matrix Heatmap:
#     # import seaborn as sns
#     # plt.figure('Normalized Confusion Matrix')
#     # sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
#     # plt.xlabel('Predicted')
#     # plt.ylabel('Actual')
#     # plt.title('Normalized Confusion Matrix')
#     # plt.show()
#
#     ## Accuracy Score:
#     from sklearn.metrics import accuracy_score
#     acc=accuracy_score(y_true=y_test,y_pred=y_pred)
#     acc_avg.append(acc)
#     print(f'Accuracy: {acc}',end='\n\n')
#
#     ## Recall or Sensitivity:
#     from sklearn.metrics import recall_score
#     recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1])
#     recall_avg.append(recall)
#     print(f'Recall: {recall}',end='\n\n')
#
#     ## Precision:
#     from sklearn.metrics import precision_score
#     precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
#     precision_avg.append(precision)
#     print(f'Precision: {precision}',end='\n\n')
#
#     ## Specificity:
#     specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
#     specificity_avg.append(specificity)
#     print(f'Specificity: {specificity}',end='\n\n')
#
#     ## F1 Score (F-Measure):
#     from sklearn.metrics import f1_score
#     f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
#     f1_avg.append(f1)
#     print(f'F1: {f1}',end='\n\n')
#
#     ## AUC (Area Under Curve):
#     from sklearn.metrics import roc_auc_score
#     auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
#     auc_avg.append(auc)
#     print(f'AUC: {auc}',end='\n\n')
#
#     ## ROC (Receiver Operation Characteristic Curve):
#     # from sklearn.metrics import RocCurveDisplay
#     #
#     # ## --- ROC with prediction method:
#     # RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
#     # plt.title('ROC Curve (From Prediction Method)')
#     # plt.show()
#     # ## --- ROC with estimation method:
#     # RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
#     # plt.title('ROC Curve (From Estimation Method)')
#     # plt.show()
#     # ## --- ROC with General Method:
#     # from sklearn.metrics import roc_curve
#     # fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
#     # plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
#     # plt.legend()
#     # plt.xlabel('False Positive Rate')
#     # plt.ylabel('True Positive Rate')
#     # plt.title('ROC')
#     # plt.grid()
#     # plt.show()
#     #
#     #
#     # #######################################
#     # ## Visualization of model's decision boundaries:
#     # #######################################
#     #
#     # model_name="SVM"
#     # is_scaled=True
#     #
#     # # --- train
#     # from matplotlib.colors import ListedColormap
#     # cmap=ListedColormap(["red","green"])
#     #
#     # x_set,y_set=x_train,y_train
#     #
#     # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#     #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#     #
#     # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
#     # plt.xlim(x1.min(),x1.max())
#     # plt.ylim(x2.min(),x2.max())
#     # for i, j in enumerate(np.unique(y_set)):
#     #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
#     # plt.title(f"{model_name} (Training set)")
#     # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
#     # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
#     # plt.legend()
#     # plt.show()
#     #
#     # # --- test
#     # from matplotlib.colors import ListedColormap
#     # cmap=ListedColormap(["red","green"])
#     #
#     # x_set,y_set=x_test,y_test
#     #
#     # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#     #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#     #
#     # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
#     # plt.xlim(x1.min(),x1.max())
#     # plt.ylim(x2.min(),x2.max())
#     # for i, j in enumerate(np.unique(y_set)):
#     #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
#     # plt.title(f"{model_name} (Test set)")
#     # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
#     # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
#     # plt.legend()
#     # plt.show()
# print('====================================================================',end='\n\n')
# print(f'Accuracy Mean: {np.mean(acc_avg)}')
# print(f'Recall Mean: {np.mean(recall_avg)}')
# print(f'Precision Mean: {np.mean(precision_avg)}')
# print(f'Specificity Mean: {np.mean(specificity_avg)}')
# print(f'F1 Mean: {np.mean(f1_avg)}')
# print(f'AUC Mean: {np.mean(auc_avg)}')









'''P02-11-Multi-Class Classification'''


## 01 iris dataset:

'''To this far , we analyzed the kind of dataset which had only two class (or target or label). We call them "Binary Class". We know 
Iris dataset from past semesters in python course and etc. The target value in Iris dataset is more than 2 (it's actualy 3), so it's "Multi-Class".'''


## 02 multi-class classification:
'''this video described how Confusion Matrix for multi-class target value. we should take one its rows (i.e.: actuals) as postive state, and negative 
state for rest of it. now it looks like a binary class confusion matrix. Note that ROC curve and AUC are NOT defined for multi-class label value.
when we're going to calculate some scores like F1, Recall ,Precision , we should note that change our label order from [0,1] , it was for 
binary class. also we must change <<<average = >>> to 'weighted' or 'macro' , because it's not a binary class anymore '''



# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# from sklearn.datasets import load_iris ## the scikit-learn module has some famous dataset like Iris in itself, so we can just import it instead
# # of using pandas module.
#
# x,y=load_iris(return_X_y=True) # input and output defined here.
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
#
#
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Support Vector Machine (SVM):
# from sklearn.svm import SVC
# model=SVC(kernel='rbf',C=10,probability=True) ## For example, we can select our kernel from:
# ## <<<kernel="rbf">>> (i.e: Radio Base Function [this is Gaussian])
# ## <<<kernel="linear">>> (i.e: linear model)
# ## <<<kernel="poly">>> (i.e: polynomial model ). In this case also we must set <<<degree =  >>>
#
# ## If we want a predict_prob in Prediction section, we should set <<<probability=True>>>
#
# ## <<<C = >>> is for penalty which discussed in intro and begining of the code.
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1,2] # Iris dataset labels w.r.t. alphabet order correspond to "Setosa" , "Versicolor" , "Virginica"
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=[0,1],average='weighted')  ## !!! --- NOTE --- !!! : Don't forget to change labels
#                                                                                   ## AND <<<average= 'weighted' >>> or <<<average= 'macro' >>>
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'Precision: {precision}',end='\n\n')
#
# # ## Specificity:
# # specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# # print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'F1: {f1}',end='\n\n')
#
# ## Classification Report:       ## we can calculate it to give us a fully report about model's evaluation
# from sklearn.metrics import classification_report
#
#
# print('Classification Report: \n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders),end='\n\n')
#
# print('Classification Report: (with named labels)\n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders,
#                                                                            target_names=['Setosa','Versicolor','Virginica']),end='\n\n')
#
#
#
#                                                       ## !! -- NOTE -- !! : ## Multi-classes target doesn't need ROC and AUC
#
# # ## AUC (Area Under Curve):
# # from sklearn.metrics import roc_auc_score
# # auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# # print(f'AUC: {auc}',end='\n\n')
# #
# # ## ROC (Receiver Operation Characteristic Curve):
# # from sklearn.metrics import RocCurveDisplay
# #
# # ## --- ROC with prediction method:
# # RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# # plt.title('ROC Curve (From Prediction Method)')
# # plt.show()
# # ## --- ROC with estimation method:
# # RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# # plt.title('ROC Curve (From Estimation Method)')
# # plt.show()
# # ## --- ROC with General Method:
# # from sklearn.metrics import roc_curve
# # fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# # plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# # plt.legend()
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC')
# # plt.grid()
# # plt.show()




'''P02-12-Predicting brand new data'''

## 01-predicting x_new:

'''Consider you completely built up a model an did prediction and evaluation. Now what if want to label new data? 
a new data gave to us (may just a 1D array) and we want to label it. see the example:'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\01-Ad.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
#
# x=df.iloc[:,3:-1].values #input
# y=df.iloc[:,-1] #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Naive-Bayes Model:
# from sklearn.ensemble import RandomForestClassifier
# model=RandomForestClassifier(n_estimators=100,max_depth=3)
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1]
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'Precision: {precision}',end='\n\n')
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders)
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# from sklearn.metrics import roc_auc_score
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# print(f'AUC: {auc}',end='\n\n')
#
# ## Classification Report:       ## we can calculate it to give us a fully report about model's evaluation
# from sklearn.metrics import classification_report
#
# print('Classification Report: \n\n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders),end='\n\n')
#
# print('Classification Report: (with named labels)\n\n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders,
#                                                                            target_names=['Non-Buyer','Buyer']),end='\n\n')
#
#
# ## ROC (Receiver Operation Characteristic Curve):
# from sklearn.metrics import RocCurveDisplay
#
# ## --- ROC with prediction method:
# RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# plt.title('ROC Curve (From Prediction Method)')
# plt.show()
#
# ## --- ROC with estimation method:
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.title('ROC Curve (From Estimation Method)')
# plt.show()
#
# ## --- ROC with General Method:
# from sklearn.metrics import roc_curve
# fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# plt.legend()
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.grid()
# plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# model_name="Naive-Bayes"
# is_scaled=True
#
# # --- train
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_train,y_train
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Training set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
# # --- test
# from matplotlib.colors import ListedColormap
# cmap=ListedColormap(["red","green"])
#
# x_set,y_set=x_test,y_test
#
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#
# plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# plt.title(f"{model_name} (Test set)")
# plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# plt.legend()
# plt.show()
#
#
# ###########################
# ## Predicting new data:
# ###########################
#
# x_new = eval(input('Enter the new data as a list to prediction: \n'))
# x_new=np.array(x_new).reshape(1,-1) ## we should reshape it as a 1 x N matrix.
# x_new=scaler.transform(x_new) ## then we MUST scale the new input data again.
# y_new_pred=model.predict(x_new)
# dic={0:'Non-Buyer',1:'Buyer'}
# print(f'Probably He\'s / She\'s {dic[int(y_new_pred)]}.')



'''let's practice it with Iris Dataset'''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\03-iris.csv')
# print(df.info(),end='\n\n')
# # print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# print('DF:\n',df,end='\n\n')
#
#
#
# x=df.iloc[:,:-1].values #input
# y=df.iloc[:,-1].values #output
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40,stratify=y)
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Naive-Bayes Model:
# from sklearn.svm import SVC
# model=SVC(C=8,kernel='rbf',probability=True)
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
# ## Evaluation:
# label_orders=[0,1,2] ## correspond to Setosa ,Versicolor ,Virginica
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'}
#             ,xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'Precision: {precision}',end='\n\n')
#
# # ## Specificity:
# # specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# # print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# # from sklearn.metrics import roc_auc_score
# # auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# # print(f'AUC: {auc}',end='\n\n')
#
# ## Classification Report:       ## we can calculate it to give us a fully report about model's evaluation
# from sklearn.metrics import classification_report
#
# print('Classification Report: \n\n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders),end='\n\n')
#
# print('Classification Report: (with named labels)\n\n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders,
#                                                                            target_names=['Setosa','Versicolor','Virginica']),end='\n\n')
#
#
# ## ROC (Receiver Operation Characteristic Curve):
# # from sklearn.metrics import RocCurveDisplay
# #
# # ## --- ROC with prediction method:
# # RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# # plt.title('ROC Curve (From Prediction Method)')
# # plt.show()
# #
# # ## --- ROC with estimation method:
# # RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# # plt.title('ROC Curve (From Estimation Method)')
# # plt.show()
# #
# # ## --- ROC with General Method:
# # from sklearn.metrics import roc_curve
# # fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# # plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# # plt.legend()
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC')
# # plt.grid()
# # plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# # model_name="SVM"
# # is_scaled=True
# #
# # # --- train
# # from matplotlib.colors import ListedColormap
# # cmap=ListedColormap(["red","green"])
# #
# # x_set,y_set=x_train,y_train
# #
# # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
# #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
# #
# # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# # plt.xlim(x1.min(),x1.max())
# # plt.ylim(x2.min(),x2.max())
# # for i, j in enumerate(np.unique(y_set)):
# #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# # plt.title(f"{model_name} (Training set)")
# # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# # plt.legend()
# # plt.show()
# #
# # # --- test
# # from matplotlib.colors import ListedColormap
# # cmap=ListedColormap(["red","green"])
# #
# # x_set,y_set=x_test,y_test
# #
# # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
# #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
# #
# # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# # plt.xlim(x1.min(),x1.max())
# # plt.ylim(x2.min(),x2.max())
# # for i, j in enumerate(np.unique(y_set)):
# #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# # plt.title(f"{model_name} (Test set)")
# # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# # plt.legend()
# # plt.show()
#
# ###########################
# ## Predicting new data:
# ###########################
#
# x_new = eval(input('Enter the new data as a list to prediction: \n'
#                    '(The Order is like this: [sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)]) \n'))
# x_new=np.array(x_new).reshape(1,-1) ## we should reshape it as a 1 x N matrix.
# x_new=scaler.transform(x_new) ## then we MUST scale the new input data again.
# y_new_pred=model.predict(x_new)
# dic={0:'Setosa',1:'Versicolor',2:'Virginica'}
# print(f'Probably it\'s {dic[int(y_new_pred)]}.')








'''P03-00-MPG dataset'''

## 01 MPG dataset:

'''Miles per Gallon Dataset has been explained in this video.'''




'''P03-01-Simple Linear Regression'''

## 01 Regression:

'''To this far, The target value of our dataset was Binary Class or Multi-Class Classification, and the model's duty finally goes toward predicting these
value by "labeling" them. so the target values were kind of subset of Categoritic kind. thus it was a "Classification" proble.
But what if the target values's column be a Numeric-based? Imagine we wanna "Estimate" fuel consumption of a new model car by information gathered from
MPG dataset. we use Regression's models here to estimate new input data. no more labeling!. We call this kind of problem: "Regression". 
 In this video a 'Simple' linear regression has been explained. simple means we "only consider one feature as x_train" data. here, y_train was fuel consupumtion.
General formula of linear regression is : Y = betta0 + betta1X . betta0 is "Intercept" or in Farsi "Arz az Mabda" , betta1 is "Slope" or in python called:
"coefficient".  
Watch the code: '''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\04-MPG.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # print('DF:\n',df,end='\n\n')
#
#
#
# x=df.iloc[:,4].values.reshape(-1,1) #input , Horsepower
# y=df.iloc[:,1].values.reshape(-1,1) #output , Fuel Consumption
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)   # !!! -- NOTE -- !!! : it's not a Classification problem, we do not
# # need to Stratify the output.
#
# ## Scaling:                 ## in this problem , we have only one feature (i.e.: Horsepower) and our data is not enormous so it's optional.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Linear Regression Model:
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
# # y_pred_prob=model.predict_proba(x_test)
# print("Regression's Line Interception: ",model.intercept_,end="\n\n")  ## Interception of linear regression's model
# print("Regression's Line Slope or Coeff: ",model.coef_,end="\n\n") ## Slope or Coefficient of linear regression's model
# print("y_pred: \n",y_pred,end="\n\n")


                                                    ## Evaluation will be discussed in next video

#
# ## Evaluation:
# label_orders=[0,1,2] ## correspond to Setosa ,Versicolor ,Virginica
#
# ## -- Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_orders)
# cm_df=pd.DataFrame(cm,index=label_orders,columns=label_orders)
# print('Confusion Matrix:\n',cm_df,end='\n\n')
#
# ## -- Normalized Confusion Matrix:
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,1))
# normalized_cm_df=pd.DataFrame(normalized_cm,index=label_orders,columns=label_orders)
# print('Normalized Confusion Matrix:\n',normalized_cm_df,end='\n\n')
#
# ## -- -- Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# ## -- -- Normalized Confusion Matrix Heatmap:
# import seaborn as sns
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'}
#             ,xticklabels=label_orders,yticklabels=label_orders)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Normalized Confusion Matrix')
# plt.show()
#
# ## Accuracy Score:
# from sklearn.metrics import accuracy_score
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
# print(f'Accuracy: {acc}',end='\n\n')
#
# ## Recall or Sensitivity:
# from sklearn.metrics import recall_score
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'Recall: {recall}',end='\n\n')
#
# ## Precision:
# from sklearn.metrics import precision_score
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'Precision: {precision}',end='\n\n')
#
# # ## Specificity:
# # specificity=cm[0,0]/np.sum(cm[0,:])  # TN/(TN+FP)
# # print(f'Specificity: {specificity}',end='\n\n')
#
# ## F1 Score (F-Measure):
# from sklearn.metrics import f1_score
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_orders,average='weighted')
# print(f'F1: {f1}',end='\n\n')
#
# ## AUC (Area Under Curve):
# # from sklearn.metrics import roc_auc_score
# # auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
# # print(f'AUC: {auc}',end='\n\n')
#
# ## Classification Report:       ## we can calculate it to give us a fully report about model's evaluation
# from sklearn.metrics import classification_report
#
# print('Classification Report: \n\n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders),end='\n\n')
#
# print('Classification Report: (with named labels)\n\n',classification_report(y_true=y_test,y_pred=y_pred,labels=label_orders,
#                                                                            target_names=['Setosa','Versicolor','Virginica']),end='\n\n')
#
#
# ## ROC (Receiver Operation Characteristic Curve):
# # from sklearn.metrics import RocCurveDisplay
# #
# # ## --- ROC with prediction method:
# # RocCurveDisplay.from_predictions(y_true=y_test,y_pred=y_pred_prob[:,1])
# # plt.title('ROC Curve (From Prediction Method)')
# # plt.show()
# #
# # ## --- ROC with estimation method:
# # RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# # plt.title('ROC Curve (From Estimation Method)')
# # plt.show()
# #
# # ## --- ROC with General Method:
# # from sklearn.metrics import roc_curve
# # fpr,tpr,threshold=roc_curve(y_true=y_test,y_score=y_pred_prob[:,1])
# # plt.plot(fpr,tpr,c='springgreen',label=f'ROC Curve. AUC= {np.round(auc,2)}')
# # plt.legend()
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC')
# # plt.grid()
# # plt.show()
#
#
# #######################################
# ## Visualization of model's decision boundaries:
# #######################################
#
# # model_name="SVM"
# # is_scaled=True
# #
# # # --- train
# # from matplotlib.colors import ListedColormap
# # cmap=ListedColormap(["red","green"])
# #
# # x_set,y_set=x_train,y_train
# #
# # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
# #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
# #
# # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# # plt.xlim(x1.min(),x1.max())
# # plt.ylim(x2.min(),x2.max())
# # for i, j in enumerate(np.unique(y_set)):
# #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# # plt.title(f"{model_name} (Training set)")
# # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# # plt.legend()
# # plt.show()
# #
# # # --- test
# # from matplotlib.colors import ListedColormap
# # cmap=ListedColormap(["red","green"])
# #
# # x_set,y_set=x_test,y_test
# #
# # x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
# #                   np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
# #
# # plt.contourf(x1,x2,model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.25,cmap=cmap)
# # plt.xlim(x1.min(),x1.max())
# # plt.ylim(x2.min(),x2.max())
# # for i, j in enumerate(np.unique(y_set)):
# #     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], s=20, color=cmap(i), label=j)
# # plt.title(f"{model_name} (Test set)")
# # plt.xlabel("Age (Scaled)" if is_scaled else "Age")
# # plt.ylabel("Estimated Salary (Scaled)" if is_scaled else "Estimated Salary")
# # plt.legend()
# # plt.show()
#
# ###########################
# ## Predicting new data:
# ###########################
#
# x_new = eval(input('Enter the new data as a list to prediction: \n'
#                    '(The Order is like this: [sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)]) \n'))
# x_new=np.array(x_new).reshape(1,-1) ## we should reshape it as a 1 x N matrix.
# x_new=scaler.transform(x_new) ## then we MUST scale the new input data again.
# y_new_pred=model.predict(x_new)
# dic={0:'Setosa',1:'Versicolor',2:'Virginica'}
# print(f'Probably it\'s {dic[int(y_new_pred)]}.')



'''P03-02-Prediction and Evaluation'''

## 01 Regression Evaluation-MAE,MSE,RMSE,MAPE:

'''In this video, some evaluation methods for regression problems has been investigated. Mean Absolute Error (MAE) , Mean Squared Error (MSE) ,
Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).
MAE = SUM_i(| y_true_i   -  y_pred_i  |) / n
MSE = SUM_i( (| y_true_i   -  y_pred_i  |)**2 ) / n
RMSE = sqrt( SUM_i ( (| y_true_i   -  y_pred_i |)**2 ) / n )
MAPE = SUM_i( (| (y_true_i   -  y_pred_i)/y_true_i | ) ) / n  '''



## 02 Regression Evaluation-r2:


'''In this video r^2 or Coefficient of Determinant has been explained. r2 implies that how much our new model got better performance and operation in 
 comparison of the ex model. usually ,in the first (i.e.: ex) model , y_pred is mean of y_true; then for first model , calculate  summation of 
 squared residual for total of data (SSTO) which is equaled to :  sum_i( e_i**2 ) and actually:
 
 SSTO = Sum_i(  |y_true_i  -  Mean_y|**2  )
  
  The Second model or next model we actually calculate sum_i( e_i**2 ) again , but y_pred is not mean of y anymore. they are themselves. hence:
  
 SSR = Sum_i(  |y_true_i  -  y_pred|**2  )
 
 Now if we compute 1 - SSR/SSTO , it would tell us how much our model got better compared to ex model, and call it r2 or coefficient of determinant model.
 
 r2 could be in (-inf , 1) interval. if it limits to 1 , means it got 100% got better, if it limits to 0, means no improvement gatthered compared to 
 ex model (i.e: y_pred == mean_y) , and if r2 < 0 , means the new model is worse than the first (i.e: y_pred == mean_y) model! 
 '''




## 03 Regression Evaluation-Conclusion:

'''CheatSheet and abstract of what have said about regression problem's evaluation.'''



## 04 Regression Evaluation-Python:

'''Now let's evaluate our linear model on MPG dataset.'''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\04-MPG.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # print('DF:\n',df,end='\n\n')
#
#
#
# x=df.iloc[:,4].values.reshape(-1,1) #input , Horsepower
# y=df.iloc[:,1].values.reshape(-1,1) #output , Fuel Consumption
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)   # !!! -- NOTE -- !!! : it's not a Classification problem, we do not
# # need to Stratify the output.
#
# ## Scaling:                 ## in this problem , we have only one feature (i.e.: Horsepower) and our data is not enormous so it's optional.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Linear Regression Model:
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
#
# print("Regression's Line Interception: ",model.intercept_,end="\n\n")  ## Interception of linear regression's model
# print("Regression's Line Slope or Coeff: ",model.coef_,end="\n\n") ## Slope or Coefficient of linear regression's model
# # print("y_pred: \n",y_pred,end="\n\n")
#
# ## Evaluation:
# from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
#
# ##--MAE:
# mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
# print("MAE: ",mae,end="\n\n")
#
# ##--MSE:
# mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
# print("MSE: ",mse,end="\n\n")
#
# ##--MAPE:
# mape=mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred)
# print("MAPE: ",mape,end="\n\n")
#
# ##--r2:
# r2=r2_score(y_true=y_test,y_pred=y_pred)
# print("r2: ",r2,end="\n\n")
#
#
# #########################
# ## Visualization :
# #########################
# plt.scatter(y_pred,y_test,c='blueviolet',ec='springgreen',label='Y-True w.r.t. Y-Predicted') ## we plotted y_test w.r.t. y_pred
# plt.plot(np.linspace(0,30,1000),np.linspace(0,30,1000),label='Y-True=Y-Pred',c='deeppink')
# plt.legend()
# plt.xlabel('y_pred')
# plt.ylabel('y_true')
# plt.show()








'''P03-03-Multiple Linear Regression'''

## 01 Multiple Linear Regression:


'''To this far , we only consider one feature for a regression problem and we called them simple regression. and because using linear regression model, 
 it was called simple linear regression. Now what if more than one feature affects our model?
(e.g.: consider MPG data set for estimating fuel consumption.) We call them Multiple Linear regression problem. now let's code. '''



# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\04-MPG.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # print('DF:\n',df,end='\n\n')
#
#
#
#
# y=df.iloc[:,1].values.reshape(-1,1) #output , Fuel Consumption
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Encoding 'origin' column with OneHotEncoder.
# from sklearn.preprocessing import OneHotEncoder
# encoded_origin=OneHotEncoder(drop='first',sparse_output=False)
# transformed_origin=encoded_origin.fit_transform(df['origin'].values.reshape(-1,1))
#
# x=np.concatenate((df.iloc[:,2:-2].values,transformed_origin),axis=1) #input
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)   # !!! -- NOTE -- !!! : it's not a Classification problem, we do not
# # need to Stratify the output.
#
# ## Scaling:                 ## in this problem , we have only one feature (i.e.: Horsepower) and our data is not enormous so it's optional.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Linear Regression Model:
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
#
# print("Regression's Line Interception: ",model.intercept_,end="\n\n")  ## Interception of linear regression's model
# print("Regression's Line Slope or Coeff: ",model.coef_,end="\n\n") ## Slope or Coefficient of linear regression's model
# # print("y_pred: \n",y_pred,end="\n\n")
#
# ## Evaluation:
# from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
#
# ##--MAE:
# mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
# print("MAE: ",mae,end="\n\n")
#
# ##--MSE:
# mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
# print("MSE: ",mse,end="\n\n")
#
# ##--MAPE:
# mape=mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred)
# print("MAPE: ",mape,end="\n\n")
#
# ##--r2:
# r2=r2_score(y_true=y_test,y_pred=y_pred)
# print("r2: ",r2,end="\n\n")
#
#
# #########################
# ## Visualization :
# #########################
# plt.scatter(y_pred,y_test,c='blueviolet',ec='springgreen',label='Y-True w.r.t. Y-Predicted') ## we plotted y_test w.r.t. y_pred
# plt.plot(np.linspace(2,26,1000),np.linspace(2,26,1000),label='Y-True=Y-Pred',c='deeppink')
# plt.legend()
# plt.xlabel('y_pred')
# plt.ylabel('y_true')
# plt.show()









'''P03-04-Polynomial Regression'''

## 01 Polynomial Regression:

'''Sometimes in regression problems (Simple or Multiple regression),we intuitively find out that plotted target value with respect to input data ,
 doesn't follow a linearly pattern. it's a degree 2 or more line fit. in other word, regession's line follows the Y = beta0 + beta1X + betaX**2 + ...  .
 when this happened , we can estimate and predict output with "LINEAR" regression model again by a simple transformation in X's and change the 
 "Simple Polynomial Regression" to "Multiple Linear Regression". for instance, change X to W1 , X2 to W2 and etc. Even we can 
 "Multiple Polynomial Regression" to "Multiple Linear Regression" too. See the example of MPG dataset again. '''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\04-MPG.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # print('DF:\n',df,end='\n\n')
#
#
#
#
# y=df.iloc[:,1].values.reshape(-1,1) #output , Fuel Consumption
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Encoding 'origin' column with OneHotEncoder.
# from sklearn.preprocessing import OneHotEncoder
# encoded_origin=OneHotEncoder(drop='first',sparse_output=False)
# transformed_origin=encoded_origin.fit_transform(df['origin'].values.reshape(-1,1))
#
# x=np.concatenate((df.iloc[:,2:-2].values,transformed_origin),axis=1) #input
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)   # !!! -- NOTE -- !!! : it's not a Classification problem, we do not
# # need to Stratify the output.
#
# ## Scaling:                 ## in this problem , we have only one feature (i.e.: Horsepower) and our data is not enormous so it's optional.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ## Polynomial Features:
# from sklearn.preprocessing import PolynomialFeatures ## to solve the polynomial regression with LinearRegression method,
# # we must first turn each of our feature (which only would be fitted with a deg>=2 line) to multiple features.
# polynomial_features=PolynomialFeatures(degree=2)
# x_train=polynomial_features.fit_transform(x_train)
# x_test=polynomial_features.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Linear Regression Model:
# from sklearn.linear_model import LinearRegression
# model=LinearRegression()
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
#
# print("Regression's Line Interception:\n ",model.intercept_,end="\n\n")  ## Interception of linear regression's model
# print("Regression's Line Slope or Coeff:\n ",model.coef_,end="\n\n") ## Slope or Coefficient of linear regression's model
# # print("y_pred: \n",y_pred,end="\n\n")
#
# ## Evaluation:
# from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
#
# ##--MAE:
# mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
# print("MAE: ",mae,end="\n\n")
#
# ##--MSE:
# mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
# print("MSE: ",mse,end="\n\n")
#
# ##--MAPE:
# mape=mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred)
# print("MAPE: ",mape,end="\n\n")
#
# ##--r2:
# r2=r2_score(y_true=y_test,y_pred=y_pred)
# print("r2: ",r2,end="\n\n")
#
#
# #########################
# ## Visualization :
# #########################
# plt.scatter(y_pred,y_test,c='blueviolet',ec='springgreen',label='Y-True w.r.t. Y-Predicted') ## we plotted y_test w.r.t. y_pred
# # plt.plot(np.linspace(2,26,1000),np.linspace(2,26,1000),label='Y-True=Y-Pred',c='deeppink')
# plt.legend()
# plt.xlabel('y_pred')
# plt.ylabel('y_true')
# plt.show()









'''P03-05-Support Vector Regression'''

## 01 SVR:

'''it looks like SVM'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\04-MPG.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # print('DF:\n',df,end='\n\n')
#
#
#
#
# y=df.iloc[:,1].values.reshape(-1,1) #output , Fuel Consumption
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Encoding 'origin' column with OneHotEncoder.
# from sklearn.preprocessing import OneHotEncoder
# encoded_origin=OneHotEncoder(drop='first',sparse_output=False)
# transformed_origin=encoded_origin.fit_transform(df['origin'].values.reshape(-1,1))
#
# x=np.concatenate((df.iloc[:,2:-2].values,transformed_origin),axis=1) #input
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)   # !!! -- NOTE -- !!! : it's not a Classification problem, we do not
# # need to Stratify the output.
#
# ## Scaling:                 ## in this problem , we have only one feature (i.e.: Horsepower) and our data is not enormous so it's optional.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Support Vector Regression Model  (SVR):
# from sklearn.svm import SVR
# model=SVR(kernel='rbf',C=1)
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
#
#
# ## Evaluation:
# from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
#
# ##--MAE:
# mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
# print("MAE: ",mae,end="\n\n")
#
# ##--MSE:
# mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
# print("MSE: ",mse,end="\n\n")
#
# ##--MAPE:
# mape=mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred)
# print("MAPE: ",mape,end="\n\n")
#
# ##--r2:
# r2=r2_score(y_true=y_test,y_pred=y_pred)
# print("r2: ",r2,end="\n\n")
#
#
# #########################
# ## Visualization :
# #########################
# plt.scatter(y_pred,y_test,c='blueviolet',ec='springgreen',label='Y-True w.r.t. Y-Predicted') ## we plotted y_test w.r.t. y_pred
# # plt.plot(np.linspace(2,26,1000),np.linspace(2,26,1000),label='Y-True=Y-Pred',c='deeppink')
# plt.legend()
# plt.xlabel('y_pred')
# plt.ylabel('y_true')
# plt.show()






'''P03-06-Decision Tree Regression'''

## 01 Decision Tree Regression:

'''It exactly looks like Decision Tree Classifier algorithm , but when depth goes through leaf's state ,algorthms calculate the mean_y and return it 
 instead a binary-class or multi-class classification.
                                
                                "CART" : (Classification And Regression Trees)  is another name of Decision Trees algorthms which dedicate to
                                 classification and regression probleem of this algorithm. '''



# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\04-MPG.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # print('DF:\n',df,end='\n\n')
#
#
#
#
# y=df.iloc[:,1].values.reshape(-1,1) #output , Fuel Consumption
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Encoding 'origin' column with OneHotEncoder.
# from sklearn.preprocessing import OneHotEncoder
# encoded_origin=OneHotEncoder(drop='first',sparse_output=False)
# transformed_origin=encoded_origin.fit_transform(df['origin'].values.reshape(-1,1))
#
# x=np.concatenate((df.iloc[:,2:-2].values,transformed_origin),axis=1) #input
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)   # !!! -- NOTE -- !!! : it's not a Classification problem, we do not
# # need to Stratify the output.
#
# ## Scaling:                 ## in this problem , we have only one feature (i.e.: Horsepower) and our data is not enormous so it's optional.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Decision Tree Regressor :
# from sklearn.tree import DecisionTreeRegressor
# model=DecisionTreeRegressor(max_depth=None)
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
#
#
# ## Evaluation:
# from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
#
# ##--MAE:
# mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
# print("MAE: ",mae,end="\n\n")
#
# ##--MSE:
# mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
# print("MSE: ",mse,end="\n\n")
#
# ##--MAPE:
# mape=mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred)
# print("MAPE: ",mape,end="\n\n")
#
# ##--r2:
# r2=r2_score(y_true=y_test,y_pred=y_pred)
# print("r2: ",r2,end="\n\n")
#
#
# #########################
# ## Visualization :
# #########################
# plt.scatter(y_pred,y_test,c='blueviolet',ec='springgreen',label='Y-True w.r.t. Y-Predicted') ## we plotted y_test w.r.t. y_pred
# # plt.plot(np.linspace(2,26,1000),np.linspace(2,26,1000),label='Y-True=Y-Pred',c='deeppink')
# plt.legend()
# plt.xlabel('y_pred')
# plt.ylabel('y_true')
# plt.show()







'''P03-07-Random Forest Regression'''

## 01 Random Forest Regression:

'''It exactly looks like Random Forest Classifier algorithm , but when n_estimator (K-trees) make their own decision at last
,algorthms calculate the mean_y of their decision and return it .'''




# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\04-MPG.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # print('DF:\n',df,end='\n\n')
#
#
#
#
# y=df.iloc[:,1].values.reshape(-1,1) #output , Fuel Consumption
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## Encoding 'origin' column with OneHotEncoder.
# from sklearn.preprocessing import OneHotEncoder
# encoded_origin=OneHotEncoder(drop='first',sparse_output=False)
# transformed_origin=encoded_origin.fit_transform(df['origin'].values.reshape(-1,1))
#
# x=np.concatenate((df.iloc[:,2:-2].values,transformed_origin),axis=1) #input
#
# ## Train Test Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=40)   # !!! -- NOTE -- !!! : it's not a Classification problem, we do not
# # need to Stratify the output.
#
# ## Scaling:                 ## in this problem , we have only one feature (i.e.: Horsepower) and our data is not enormous so it's optional.
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## : Random Forest Regressor:
# from sklearn.ensemble import RandomForestRegressor
# model=RandomForestRegressor(n_estimators=100,max_depth=None)
# model.fit(x_train,y_train)
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# y_pred=model.predict(x_test)
#
#
# ## Evaluation:
# from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score
#
# ##--MAE:
# mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
# print("MAE: ",mae,end="\n\n")
#
# ##--MSE:
# mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
# print("MSE: ",mse,end="\n\n")
#
# ##--MAPE:
# mape=mean_absolute_percentage_error(y_true=y_test,y_pred=y_pred)
# print("MAPE: ",mape,end="\n\n")
#
# ##--r2:
# r2=r2_score(y_true=y_test,y_pred=y_pred)
# print("r2: ",r2,end="\n\n")
#
#
# #########################
# ## Visualization :
# #########################
# plt.scatter(y_pred,y_test,c='blueviolet',ec='springgreen',label='Y-True w.r.t. Y-Predicted') ## we plotted y_test w.r.t. y_pred
# # plt.plot(np.linspace(2,26,1000),np.linspace(2,26,1000),label='Y-True=Y-Pred',c='deeppink')
# plt.legend()
# plt.xlabel('y_pred')
# plt.ylabel('y_true')
# plt.show()











'''P04-01-Prerequisites + Clustering Concept'''

## 01-Distance:

'''In this video Euclidean , Manhattan and Minkowski Distance has been explained.
  consider: p=[p1,...,pk]
            q=[q1,...,qk]
            
  the Euclidean Norm or distance between p and q: 
            d(p,q) = sqrt( SUM_i (p_i - q_i)**2 )
  the Manhattan Norm or distance between p and q:
            d(p,q) = SUM_i |p_i - q_i|   
  the Minkowski Norm or distance between p and q:
            d(p,q) = ( SUM_i |p_i - q_i|**p )**(1/p)
            Which: 
                    if p==1 : it's Manhattan distance
                    if p==2 : it's Euclidean distance'''




## 02 Center:

'''In this video , the method for calculation of Center point has been explained.'''



## 03 Clustering Concept:

'''Clustering Concept has been described in this video. To this far, our problems were Supervised Learning, that means we only depends on
kind of Y column (i.e.: output ), whether it's categorical (binary or multi-class) for classification or numeric for regression , tried to solve 
the problem. while in Unsupervised Learning there is no y_column; so there is neither labels nor estimation. here, we only have features as X columns,
and should cluster our features to find a conclusion for our problem.'''







'''P04-02-K-Means'''

## 01 K-Means:

'''In this video ,"K-Means", one of famous algorthm which performs in clustering concept in unsupervised learning is explained.
 First,we should know our data.As we know, one of method for knowing data is visualiztion.
 we can visualized our data if we have two or three features with scatter plot to find out how many cluster we need or we want.
if features would be more, we should try others methods.
Secondly , we select K, as number of cluster we want. then the K-Means algorthm select K points randomly. (or deterministicly if we set <<<init=k-means++>>>.
it's better to do this to avoid error in clustering in final stage.) then the algorthm comes and calculate distance metric between each scattered point of
our features AND these random or deterministic points which set lastly.algorithm compares these distance metric of each scattered point with those
random/deter points and will labels scattered points which are nearer to that specific random/deter point. Now comes select new K random/deter points 
based on clustered data recently and calculate distance metric of those new K random/deter points, and update labeling our scattered data
 the way which are nearer to K random/deter. if in updating process the clusters change , we should repeat and select new K-random/deter points based on
 updated cluster and repeat the algorthm. if clusters doesn't change , we conclude our job! 
 For better understanding ,Consider the Customers data set: 
'''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\05-Customers.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # # print('DF:\n',df,end='\n\n')
#
#
# x=df.iloc[:,:].values.reshape(-1,2) #input
#
# ## we don't have y column. we actually know that!
#
# ##plotting scatter data
# plt.scatter(x[:,0],x[:,1],fc='blueviolet',ec='springgreen')
# plt.xlabel('Annual Income')
# plt.ylabel('Spending Score')
# plt.title('Customers Dataset')
# plt.show()  ## we here, find out k==5 is a good idea
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## In Clustering, of course we don't have smth like Train Test Splitting.
#
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x=scaler.fit_transform(x)
#
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## K-Means with K-Means++ initialization:
# from sklearn.cluster import KMeans
# model=KMeans(n_clusters=5,init="k-means++") ## we chose deterministic selecting k points
# model.fit(x)    # !!! --- NOTE --- !!! : we can comment this line and jump to prediction section and use <<<model.fit_predict(x)>>>
#
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
# #y_pred=model.fit_predict(x)
# y_pred=model.predict(x)                 #or:
# # y_pred=model.labels_
# print('y_pred or Pred_labels: \n',y_pred,end='\n\n')








## 02 K-Means-Elbow:

'''In last video , as we saw , we chose k==5 ; Since we ploted the scattered features and find out k=5 is proper. But what if the features that we wotk on,
are more than 3 , and we can't plot them to find out proper k value for clustering? One of the methods which give us the chance to figure proper K out is
"Elbow" method. for a range of K, we first calculate "Within Cluster Sum of Squares" (WCSS) which known as "Inertia" in python. then we plot
WCSS with respect to range of K's. the "Elbow" point is a proper k for our clustering problem.
For instance if we consider two cluster , the WCSS is equaled to: (P are points, i are index, ISIN means being a member in a set, d(.,.) is distance metric,
and C_1 and C_2 are center of Cluster1 and Cluster2 correspondly:
 
    WCSS = Sum_P_i_ISIN_Cluster1 (d(P_i,C_1)**2)   +   Sum_P_i_ISIN_Cluster2 (d(P_i,C_2)**2)
    
The more Number of cluster (i.e.: big K value) give us the less value of WCSS or Inertia.
and in other word: 
The more Size of cluster (i.e.: small K value), The more value of WCSS or Inertia. 
'''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\05-Customers.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # # print('DF:\n',df,end='\n\n')
#
#
# x=df.iloc[:,:].values.reshape(-1,2) #input
#
# ## we don't have y column. we actually know that!
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## In Clustering, of course we don't have smth like Train Test Splitting.
#
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x=scaler.fit_transform(x)
#
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## K-Means with K-Means++ initialization:
# from sklearn.cluster import KMeans
# ## Assume that we don't know what value of K is suitable for our KMeans clustering algorithm, So we compute Inertia (i.e.: WCSS) for a range of K.
# ## (e.g.: range(2,11)) . then we plot WCSS w.r.t. K's range. the Elbow point is a suitable value for K.
#
# WCSS=[] #empty list for appending the wcss values to plot it later.
#
# for k in range(2,11):
#     model=KMeans(n_clusters=k,init="k-means++") ## we chose deterministic selecting k points
#     model.fit(x)    # !!! --- NOTE --- !!! : we can comment this line and jump to prediction section and use <<<model.fit_predict(x)>>>
#     WCSS.append(model.inertia_)
#
# ## Ploting the Inertia w.r.t. K's range:
# plt.plot(range(2,11),WCSS,marker='o',mec='springgreen',mfc='teal',color='blueviolet')
# plt.title('Inertia w.r.t. K\'s range to find Elbow Point')
# plt.xlabel('K')
# plt.ylabel('WCSS')
# plt.show()
#
# # ##############################
# # ## Prediction and Evaluation:
# # ##############################
# #
# # ## Prediction:
# # #y_pred=model.fit_predict(x)
# # y_pred=model.predict(x)                 #or:
# # # y_pred=model.labels_
# # print('y_pred or Pred_labels: \n',y_pred,end='\n\n')








'''P04-03-Prediction and Evaluation'''

## 01-Silhouette:

'''In unsupervised learning and of course clustering problem, since we don't have a target value (i.e.: y column) , it's always chellenging for us
 to evaluate our model. one of evaluation metric score is "Silhouette" score. 
 Note:
            -1<Silhoutte<1 
    the closer sillhoutte score to 1, the better clustering model 1.'''



# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\05-Customers.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # # print('DF:\n',df,end='\n\n')
#
#
# x=df.iloc[:,:].values.reshape(-1,2) #input
#
# # we don't have y column. we actually know that!
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## In Clustering, of course we don't have smth like Train Test Splitting.
#
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x=scaler.fit_transform(x)
#
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## K-Means with K-Means++ initialization:
# from sklearn.cluster import KMeans
# ## Assume that we don't know what value of K is suitable for our KMeans clustering algorithm, So we compute Inertia (i.e.: WCSS) for a range of K.
# ## (e.g.: range(2,11)) . then we plot WCSS w.r.t. K's range. the Elbow point is a suitable value for K.
#
# WCSS=[] #empty list for appending the wcss values to plot it later.
# sillhoutte=[]  #empty list for appending the wcss values to plot it later.
# for k in range(2,11):
#     model=KMeans(n_clusters=k,init="k-means++") ## we chose deterministic selecting k points
#     model.fit(x)    # !!! --- NOTE --- !!! : we can comment this line and jump to prediction section and use <<<model.fit_predict(x)>>>
#     WCSS.append(model.inertia_)
#     ##############################
#     ## Prediction and Evaluation:
#     ##############################
#
#     ## Prediction:
#     #y_pred=model.fit_predict(x)
#     y_pred=model.predict(x)                 #or:
#     # y_pred=model.labels_
#     # print('y_pred or Pred_labels: \n',y_pred,end='\n\n')
#
#     ## Evaluation:
#     from sklearn.metrics import silhouette_score
#     sil=silhouette_score(X=x,labels=y_pred)
#     sillhoutte.append(sil)
#
#     ## Visuallize and Plotting Scattered Data while clustered
#     plt.figure('Visuallize and Plotting Scattered Data while clustered')
#     plt.subplot(3,3,k-1)
#     plt.scatter(x[:,0],x[:,1],c=model.labels_)
#     plt.xlabel('Annual Income')
#     plt.ylabel('Spending Score')
#     plt.title(f'K = {k}')
#     plt.suptitle('Scattered Data while clustered')
#     plt.tight_layout()
#
#
# ## Ploting the Inertia w.r.t. K's range:
# plt.figure('Ploting the Inertia w.r.t. K\'s range')
# plt.plot(range(2,11),WCSS,marker='o',mec='springgreen',mfc='teal',color='blueviolet')
# plt.title('Inertia w.r.t. K\'s range to find Elbow Point.')
# plt.xlabel('K')
# plt.ylabel('WCSS')
#
# ## Ploting the Silhoutte Score w.r.t. K's range:
# plt.figure('Ploting the Silhoutte Score w.r.t. K\'s range')
# plt.plot(range(2,11),sillhoutte,marker='o',mec='springgreen',mfc='teal',color='blueviolet')
# plt.title('Silhoutte Score w.r.t. K\'s range.')
# plt.xlabel('K')
# plt.ylabel('Silhoutte')          # seems k==5 was our best choice
#
#
# plt.show()









'''P04-04-Hierarchical Clustering'''

## 01 Linkage Methods:

'''In this video , the distance between two arbitrary "cluster" (Not exactly the scattered data points!) with multiple method has been taught.
those method were:

        Single Linkage:
        
    Consider two arbitrary cluster which each one of them has their own number of scattered data. The Single Linkage distance defines as: 
    Minimum Distance of two clusters; i.e. distance of the two Nearest scattered data where each scattered point stands in its own cluster.
        
        Complete Linkage:
        
    Consider two arbitrary cluster which each one of them has their own number of scattered data. The Complete Linkage distance defines as: 
    Maximum Distance of two clusters; i.e. distance of the two Farthest scattered data where each scattered point stands in its own cluster.
        
        Average Linkage:
        
    Consider two arbitrary cluster which each one of them has their own number of scattered data. The Average Linkage distance defines as:
    Average Distance of two clusters; i.e. we compute distance of EACH scattered data in cluster A from all scattered data in others cluster;
    (for better understanding consider nA x nB distance matrix that nA: number of scattered data in cluster A and nB: number of scattered data in
    cluster B ) 
    
        Centroid Linkage:
        
    Consider two arbitrary cluster which each one of them has their own number of scattered data. The Centroid Linkage distance defines as:
    Computing centroid point of two arbitrrary cluster by finding average coordination gathered by scattered datas on that specific cluster, then
    calculate distance between these two average coordination.
    
        Ward Linkage:
        
    Consider two arbitrary cluster which each one of them has their own number of scattered data. The Ward Linkage distance defines as:
    Calculating the Cenroid distance (by Euclidean method),then:
                                                                1: Raise it to Power of 2
                                                                2: Multiply it to : 2(nAnB)/(nA + nB)   (Note that some refrences don't consider num "2")
                                                                3: Take the root of it'''







## 02 HC:

'''In Hierarchical Clustering, we firstly assume each scattered input data as a cluster. then calculate the distance between these clusters from
 each other (with one of those method which discussed in last video. like Single Linkage ,complete linkage, average linkage, centroid linkage and
 Ward linkage). The two nearest cluster will be merge and fuse together and will make a new cluster. in the next move, we repeat these steps and 
 continue this repeating until build a super cluster which contains all data in itself. Now we plot "Dendrogram" to give us a better visualization
 and way to decide how much cluster is needed.
 After plotting the Dendrogram (where show us the wizard of those input scattered data cluster fusion), we plot some <<axhline>> as many as the horizontal
 Dendrogram lines onto horizontal Dendrogram lines itself. Now we actually divided our Dendrogram chart to some vertical lines too.
 The Number of Vertical lines that have the longest length between all other vertical lines is the suitable value for Number of Cluster.'''





## 03 Hc-Python:

'''Now let's code about hierarchical clustering model for Customer dataset.'''


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# ##################################
# ## Reading Data:
# ##################################
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
#
# df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\05-Customers.csv')
# print(df.info(),end='\n\n')
# print('DF Head:\n',df.head(),end='\n\n')
# print('DF Shape:\n',df.shape,end='\n\n')
# # # print('DF:\n',df,end='\n\n')
#
#
# x=df.iloc[:,:].values.reshape(-1,2) #input
#
# # we don't have y column. we actually know that!
#
#
# ##############################
# ## Preprocessing:
# ##############################
#
# ## In Clustering, of course we don't have smth like Train Test Splitting.
#
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x=scaler.fit_transform(x)
#
#
#
# ##############################
# ## Building Model:
# ##############################
#
# ## Hierarchical Clustering's Linkage and Dendrogram:
# from scipy.cluster.hierarchy import linkage,dendrogram
# linkages=linkage(x,method='ward',metric='euclidean') # Ward linkage is most common between all linkage distance. # default distance metric is euclidean-based..
# dendrogram(linkages)
# plt.xticks([])
# plt.ylabel('Ward Linkage Distance')
# plt.title('Dendrogram for Hierarchical Clustering Model')
# plt.show() # seems again, 5 cluster is good choice.
#
#
# ## Hierarchical Clustering model (Agglomerative Clustering model)
# from sklearn.cluster import AgglomerativeClustering #Agglomorate means accumulate somehow, building and establish a Stack like stack of stone, wood in somewhere.
# model=AgglomerativeClustering(n_clusters=5,linkage='ward') #linkage distance method's default is Ward.
# model.fit(x)    # !!! --- NOTE --- !!! : we can comment this line and jump to prediction section and use <<<model.fit_predict(x)>>>
#
# ##############################
# ## Prediction and Evaluation:
# ##############################
#
# ## Prediction:
#
# #y_pred=model.fit_predict(x)
# # y_pred=model.predict(x)                 <<<predict>>> method doesn't exist in Agglomorative class, so <<<labels_>>> attribute or <<<fit_predict>>> method is other choices:
# y_pred=model.labels_
# # print('y_pred or Pred_labels: \n',y_pred,end='\n\n')
#
# ## Evaluation:
#
# from sklearn.metrics import silhouette_score
# sil=silhouette_score(X=x,labels=y_pred)
# print(f'Silhoutte Score: {sil}',end="\n\n")
#
#
# ## Visuallize and Plotting Scattered Data while clustered
# plt.figure('Visuallize and Plotting Scattered Data while clustered')
# plt.scatter(x[:,0],x[:,1],c=model.labels_)
# plt.xlabel('Annual Income')
# plt.ylabel('Spending Score')
# plt.title(f'K = {model.n_clusters}')
# plt.suptitle('Scattered Data while clustered')
# plt.show()







'''P05-01-Feature Selection'''


## 01 Feature Selection:

'''In this video , talked about "Feature Selection" . In a problem (e.g.: multi-class classification), some features (i.e.: columns) may not need to 
consider for fit and learning. so we shouldn't insert this column as part of input data due to diminishing of our evaluation scores. to determine that 
how a column can effect the evaluation scores , we define "Delta of " that scores (e.g.: delta of f1 , delta of recall , delta of r2, delta of MSE and 
etc.) and in python we call it "Feature Score". We once train and fit our ALL input data and compute an arbitrary score for example like F1 for them. 
once again, we eliminate one of column (i.e.: feature) and calculate that same (here F1) score. the differentiate between these two gathered score will
give us the DeltaF or feature score for F1.

 Those score that being high is a good things in them like f1, recall and etc, as their Delta increase when we eliminate a feature (e.g.: x1), 
the importancy of that feature (e.g.: that x1) will increase. if DelataF==0 , then that column (i.e.: here x1) is not important to consideration for
inserting it to fitting and it just increase the time if modeling , simulation and learning. if the DeltaF<0 , then that column (i.e.: here x1) not only
is not important to consideration , but also ruins the process and evaluation score and diminish that score (e.g.: here F1 ).

Note that if the score where we are going to compute being high value is not a good things like errors (e.g.: MSE , MAE, MAPE, RMSE ) we should multiple
the feature score with a Minsus, to not getting confused. 

we can also normalize the feature scores that we computed for our model to have a better understanding:

normalized_feature_score = (feature score) / max(abs(feature score))
'''



## 02 Feature Selection-Python:

'''We usually shuffle that specific feature (i.e.: column) instaed of eliminating it to finding out about its feature score.'''



##              !!!! ----------- Firstly, we define a function to do Cross Validation (with some modification) to us. ----------- !!!!

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# def CV(x,y): ## this function is going to K-Fold (with stratifaction) our data , then with SVC , classify our model then find and return
#     # the average of F1 score, Accuracy score, Recall and Precision Score correspondly.
#
#     f1_avg = []  ## we once, consider F1 to take average.
#     acc_avg=[]   ## we once, consider Accuracy Score to take average.
#     recall_avg=[]   ## we once, consider Recall or Sensitivity Score to take average.
#     precision_avg=[]   ## we once, consider Precision Score to take average.
#
#     for k in range(10):
#         ##################################
#         ## Cross Validation:
#         #################################
#
#         from sklearn.model_selection import StratifiedKFold
#         cv=StratifiedKFold(n_splits=5,shuffle=True)
#
#
#
#         for train_idx, test_idx in cv.split(x, y):
#             ##############################
#             ## Preprocessing:
#             ##############################
#
#
#
#             ## Data Splitting:
#             x_train,x_test=x[train_idx],x[test_idx]
#             y_train,y_test=y[train_idx],y[test_idx]
#
#             ## Scaling:
#             from sklearn.preprocessing import StandardScaler
#             scaler=StandardScaler()
#             x_train=scaler.fit_transform(x_train)
#             x_test=scaler.transform(x_test)
#
#             ##############################
#             ## Building Model:
#             ##############################
#             from sklearn.svm import SVC
#             model=SVC(kernel='rbf',probability=True,C=1.0)
#             model.fit(x_train,y_train)
#
#
#             ##############################
#             ## Prediction and Evaluation:
#             ##############################
#
#             ## Prediction:
#             y_pred=model.predict(x_test)
#             y_pred_prob=model.predict_proba(x_test)
#
#
#             ## Evaluation:
#             label_order=[0,1,2]
#
#             ## F1 Score:
#             from sklearn.metrics import f1_score
#             f1=f1_score(y_test,y_pred,labels=label_order,average='weighted')
#             f1_avg.append(f1)
#
#             ## Accuracy Score:
#             from sklearn.metrics import accuracy_score
#             acc=accuracy_score(y_test,y_pred)
#             acc_avg.append(acc)
#
#             ## Recall Score:
#             from sklearn.metrics import recall_score
#             recall=recall_score(y_test,y_pred,labels=label_order,average='weighted')
#             recall_avg.append(recall)
#
#             ## Precision Score:
#             from sklearn.metrics import precision_score
#             precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_order,average='weighted')
#             precision_avg.append(precision)
#
#     return np.mean(f1_avg),np.mean(acc_avg),np.mean(recall_avg),np.mean(precision_avg)
#
# ##             ------- Now we can use CV(x,y) function in our problem -----
#
#
# ##################################
# ## Reading Data:
# ##################################
#
# ## Defining x and y:
#
# from sklearn.datasets import load_iris
#
# X,y=load_iris(return_X_y=True) ## Note that we consider X (capital X) as input
#
#
# ##################################
# ## Feature Selection:
# ##################################
#
# f1_init,acc_init,recall_init,precision_init=CV(X,y) ## First, we compute the initial f1 average (when all features (i.e.: all x_i has been considered) exist.)
# feature_score_f1=[]
# feature_score_acc=[]
# feature_score_recall=[]
# feature_score_precision=[]
#
# for i in range(X.shape[1]): ## We are going to Shuffle each column then compare the computed F1 score with f1_initial (and other metrics like acc and... too),
#     # which will call it Feature Score.
#     x=X.copy() # take a deep copy
#     np.random.shuffle(x[:,i]) # note that shuffle function wont return an array! so do not update your array with updating that value
#     feature_score_f1.append(f1_init-CV(x,y)[0]) ## The feature score for f1. the values in each index implies that when the x_i column
#     # get eliminated or shuffled, how much the F1 score would be changed.
#     feature_score_acc.append(f1_init-CV(x,y)[1]) ## The feature score for Accuracy. the values in each index implies that when the x_i column
#     # get eliminated or shuffled, how much the Accuracy score would be changed.
#     feature_score_recall.append(f1_init-CV(x,y)[2]) ## The feature score for Recall. the values in each index implies that when the x_i column
#     # get eliminated or shuffled, how much the Recall score would be changed.
#     feature_score_precision.append(f1_init-CV(x,y)[3]) ## The feature score for Precision. the values in each index implies that when the x_i column
#     # get eliminated or shuffled, how much the Precision score would be changed.
#
# print('Feature Score for F1: \n ',feature_score_f1,end='\n\n')
# print('Feature Score for Accuracy: \n ',feature_score_acc,end='\n\n')
# print('Feature Score for Recall: \n ',feature_score_recall,end='\n\n')
# print('Feature Score for Precision: \n ',feature_score_precision,end='\n\n=======================================================\n\n')
#
# ## We can scale the feature score for better understanding:
# feature_score_scaled_f1=feature_score_f1/np.max(np.abs(feature_score_f1))
# feature_score_scaled_acc=feature_score_acc/np.max(np.abs(feature_score_acc))
# feature_score_scaled_recall=feature_score_recall/np.max(np.abs(feature_score_recall))
# feature_score_scaled_precision=feature_score_precision/np.max(np.abs(feature_score_precision))
#
# print('Scaled Feature Score for F1 rounded to 3 digits after point: \n ',np.round(feature_score_scaled_f1,3),end='\n\n')
# print('Scaled Feature Score for Accuracy rounded to 3 digits after point: \n ',np.round(feature_score_scaled_acc,3),end='\n\n')
# print('Scaled Feature Score for Recall rounded to 3 digits after point: \n ',np.round(feature_score_scaled_recall,3),end='\n\n')
# print('Scaled Feature Score for Precision rounded to 3 digits after point: \n ',np.round(feature_score_scaled_precision,3),end='\n\n')







'''P05-02-Feature Extraction'''

## 01 PCA:

'''In this video , talked about "Principal Component Analysis" (PCA) or "Feature Extraction". Actually , we can modify or reduce the dimension of
our scattered data. when we fit a hyperplane to our scatterd data (i.e.: w.r.t. x1,x2,..xn), we can name that hyperplane as PC1 and then the next
pattern that we could fit with a hyperplane PC2 and...  . For instance , for a scattered data with only 2 features (x1 and x2), when
 we fit for example a linear line to them , we can also fit another line which is orthogonal to the first line where it shows the scattering of data
 over line 1 , i.e PC1. We call it PC2. Now we can plot a new coordination PC2 w.r.t. PC1 for better and faster computation. if the range of 
 PC2 be short compared to PC1, we can neglect the PC2 axis and do a "Dimension Reduction".  
 
 Note that sometimes even dimension reduction beside to decrease the volume of computation, it can help increasing the evaluation scores too!!'''



## 02 PCA+Kernel PCA (Python):

'''To better understanding of above passage, consider the cancer dataset and let's code! '''

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# #####################################
# ## Reading Data:
# #####################################
# from sklearn.datasets import load_breast_cancer
#
# pd.set_option('display.width',None)
# pd.set_option('display.max_column',None)
#
# # df=pd.read_csv(r'D:\Koolac\06- Machine Learning\P00-01-Datasets\06-Cancer.csv')
# # x=df.iloc[:,:-1].values
# # y=df.iloc[:,-1].values
# # print(df.info(),end='\n\n')
# # print('Cancer DF Head: \n\n',df.head(),end='\n\n')
#
# x,y=load_breast_cancer(return_X_y=True,as_frame=True) #input and output as an DataFrame and Series
# print('Features of Breast Cancer Dataset:\n\n',x,end='\n\n')
# print('Target Value of Breast Cancer Dataset: (0:Benignant , 1:Malignant)\n\n',y,end='\n\n')
#
# x=x.values
# y=y.values #input and output as an array
#
#
# #####################################
# ## Preprocessing:
# #####################################
#
# ## Train-Test-Splitting:
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y,random_state=40) #it's a binary classification , it's good to stratifing the choices
#
# ## Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
#
#
# ## Feature Extraction  -> PCA :
# from sklearn.decomposition import PCA ## PCA has a linear approach, we can use KernelPCA instead of it.
# pca=PCA(n_components=None)         ## First, we find out how many Principle Components are enough , thwn we change the argument.
# x_train=pca.fit_transform(x_train)
# x_test=pca.transform(x_test)
# print('PCA Variance Ratio :\n',pca.explained_variance_ratio_,end='\n\n') #For indicating Variance and Importancy of each new feature (i.e.: PC_i)
# # compared to another, we use this command. Those features which have bigger  Variance, mean they are more vulnerable on the whole data and should
# # consider in our silulation carefully, and those features which has few variance, can be ruduct to increase the performance of algorithms and may
# # better evaluarion scores later.
#
# print('PCA Cumulative Sum Variance Ratio : \n',np.cumsum(pca.explained_variance_ratio_),end='\n\n')#For indicating a better look for variances,we took
# #CumSum of them.
#
#
#
# #####################################
# ## Building The Model:
# #####################################
# from sklearn.svm import SVC
# model=SVC(probability=True,kernel='rbf')
# model.fit(x_train,y_train)
#
#
#
# #####################################
# ## Prediction and Evaluation:
# #####################################
#
# ##Prediction:
# y_pred=model.predict(x_test)
# y_pred_prob=model.predict_proba(x_test)
#
#
# ##Evaluation:
# from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,RocCurveDisplay,classification_report
# label_order=[0,1]
#
# ## Accuracy:
# acc=accuracy_score(y_true=y_test,y_pred=y_pred)
#
# ## Recall:
# recall=recall_score(y_true=y_test,y_pred=y_pred,labels=label_order,average='binary')
#
# ## Precision:
# precision=precision_score(y_true=y_test,y_pred=y_pred,labels=label_order,average='binary')
#
# ## F1:
# f1=f1_score(y_true=y_test,y_pred=y_pred,labels=label_order,average='binary')
#
# ## AUC:
# auc=roc_auc_score(y_true=y_test,y_score=y_pred_prob[:,1])
#
# list=[acc,recall,precision,f1,auc]
# printable_list=['Accuracy','Recall','Precision','F1','AUC']
#
# for i,j in enumerate(printable_list):                       ## Printing the evaluation Scores
#     print(f'The {j} Score: {list[i]}',end='\n\n')
#
#
# ## ROC Curve:
#
# RocCurveDisplay.from_estimator(estimator=model,X=x_test,y=y_test)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC for Cancer Dataset')
#
#
# ## Confusion Matrix:
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_true=y_test,y_pred=y_pred,labels=label_order)
#
# ## Specificity:
# specificity=cm[0,0]/np.sum(cm[0,:])
# print(f'The Specificity Score: {specificity}',end='\n\n')
#
#
# ## Classification Report:
# report=classification_report(y_true=y_test,y_pred=y_pred,labels=label_order,target_names=['Benignant','Malignant'])
# print('Classification Report: \n\n',report,end='\n\n')
#
# ## Plotting Confusion Matrix:
# import seaborn as sns
# plt.figure('Confusion Matrix')
# sns.heatmap(cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_order,yticklabels=
#             label_order)
#
# plt.title('Confusion Matrix For Cancer Dataset')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
#
#
# ## Normalized Confusion Matrix:
#
# normalized_cm=cm/(np.sum(cm,axis=1).reshape(-1,2))
#
# ## Plotting Normalized Confusion Matrix:
#
# plt.figure('Normalized Confusion Matrix')
# sns.heatmap(normalized_cm,cmap='Greens',annot=True,fmt='0.2f',cbar_kws={'orientation':'vertical','label':'Color Bar'},xticklabels=label_order,yticklabels=
#             label_order)
#
# plt.title('Normalized Confusion Matrix For Cancer Dataset')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()




'''P05-03-Dimensionality Reduction'''

## 01 Dimensionality Reduction:

'''A reminding and recall about difference between "Feature Selection" and "Feature Extraction" . '''




'''P06-01-Final Notes'''

## 01 Final Notes:

'''As a remind for the whole bootcamp, the difference between Supervised and Unsupervised Learning has been explained. 
Note that the PCA method for feature extraction, is a Unsupervised Learning.'''







'''DRILL I'''

#A:

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# #Data:
#
# x = np.array([
#         [3, 1],
#         [4, 2],
#         [1, 4],
#         [2, 5],
#         [5, 5],
#         [6, 4]
#     ])
#
#
# ## We are going to clustering this data with K-Means algorithm
#
# ###########################
# # Preprocessing:
# ###########################
#
#
# #Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x=scaler.fit_transform(x)
#
#
# ###########################
# # Building The Model:
# ###########################
#
# from sklearn.cluster import KMeans
# wcss=[]
# silhouette_list=[]
# for i in range(2,x.shape[0]):
#     model=KMeans(n_clusters=i)
#     model.fit(x)
#
#     wcss.append(model.inertia_)
#
#     ###########################
#     # Prediction and Evaluation:
#     ###########################
#
#     # Prediction:
#     labels=model.predict(x)
#
#
#     # Evaluation:
#     from sklearn.metrics import silhouette_score
#     silhouette=silhouette_score(x,labels)
#     silhouette_list.append(silhouette)
#
# # Plotting Silhouette Score w.r.t. K Cluster:
# plt.plot(range(2,x.shape[0]),silhouette_list,color='k',marker='o',mfc='red',ms=10)
# plt.title(f'Best k according to Sil. is: {np.argmax(silhouette_list)+2}')
# plt.show()

'''So it seems k==3 is the best choice fur clustering our data'''

#B:

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# #Data:
#
# x = np.array([
#         [3, 1],
#         [4, 2],
#         [1, 4],
#         [2, 5],
#         [5, 5],
#         [6, 4]
#     ])
#
#
# ## We are going to clustering this data with K-Means algorithm
#
# ###########################
# # Preprocessing:
# ###########################
#
#
# #Scaling:
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# x=scaler.fit_transform(x)
#
#
# ###########################
# # Building The Model:
# ###########################
#
# from sklearn.cluster import KMeans
#
#
#
# model=KMeans(n_clusters=3)
# model.fit(x)
# ###########################
# # Prediction and Evaluation:
# ###########################
#
# # Prediction:
# labels=model.predict(x)
#
#
# # Evaluation:
# from sklearn.metrics import silhouette_score
# silhouette=silhouette_score(x,labels)
#
# # Plotting the Clustered Data:
#
# plt.scatter(x[:,0],x[:,1],c=labels)
# plt.xlabel('X 1')
# plt.ylabel('X 2')
# plt.title(f'Clustered Data, K={model.n_clusters}')
# plt.show()