import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import itertools

def find_best_fit():
    X = mydata_ref1['distance travelled with remaining fuel']
    aic = []
    param1 = []
    # warnings.filterwarnings("ignore")
    for param in pdq:
        try:
            model = ARIMA(X, order=param)
            model_fit = model.fit()
            a = model_fit.aic
            param1.append(param)
            aic.append(a)
            print('ARIMA{} - AIC:{}'.format(param, a))
        except:
            #   print("inside except")
            continue
    mini = min(aic)
    index = aic.index(mini)
    print('selscted ARIMA model is ARIMA{} - AIC:{}'.format(param1[index], mini))
    return param1[index]

mydata_ref1=pd.read_csv('dataset.csv',parse_dates=True)
#print(mydata_ref1.head())
p = range(1,21)
d = range(1,4)
q = range(0,4)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
#print(pdq)
#print(len(pdq))
pdq=(19,3,2)#pdq=find_best_fit()
#p,d,q=pdq[0],pdq[1],pdq[2]
#print(type(pdq))

# fit model
x = mydata_ref1['distance travelled with remaining fuel']
model = ARIMA(x, order=pdq)
model_fit = model.fit()
model_fit0=model_fit

# split data into train and test and forecast test
X = mydata_ref1['distance travelled with remaining fuel']
size = int(len(X)*0.6)
train=mydata_ref1['distance travelled with remaining fuel'].iloc[:size]
test=mydata_ref1['distance travelled with remaining fuel'].iloc[size:]
#print(test)
#Fit ARIMA(1,1,1) on train data and find parameters
model = ARIMA(train, order=(19,3,2))
model_fit = model.fit(start_params=model_fit0.params)
#predict train data
print('Forecasting Training Data')
pred_train = model_fit.predict(start=1,end=size)
#print(pred_train)
MSE_train = mean_squared_error(train, pred_train)
RMSE_train = sqrt(MSE_train)
#RMSPE_train = RMSPE(train,pred_train)

print('train RMSE: %.5f' % RMSE_train)
#print('train RMSPE: %.5f' % RMSPE_train)
pred_train=pred_train.tolist()
#print("pred_train",pred_train)
print('Forecasting Test Data')
# predict test data
history = [x for x in train]
pred_test = list()
for t in range(len(test)):
	model = ARIMA(history, order=(19,3,2))
	model_fit = model.fit(start_params=model_fit.params)
	output = model_fit.forecast()
	yhat = output[0]
	yhat = yhat.tolist()
	pred_test.append(yhat)
	obs = test[t+size]
	history.append(obs)
	#print('predicted=%f, expected=%f, difference=%f' % (yhat, obs,yhat-obs))
MSE_test = mean_squared_error(test, pred_test)
RMSE_test = sqrt(MSE_test)
#RMSPE_test = RMSPE(test,pred_test)
print('test RMSE: %.5f' % RMSE_test)
#print('Test RMSPE: %.5f' % RMSPE_test)
predictions = np.concatenate((pred_train,pred_test),axis=0)
#print(pred_test)
#print(len(pred_test))

# line plot of observed vs predicted

fig, ax = plt.subplots(1)
predictions_df = pd.DataFrame(pred_test,columns=['predicted'])
new_train=train
new_train=new_train.reset_index()
#print("Train",type(new_train))

print(len(predictions_df))

list1=[x for x in range(len(new_train),len(new_train)+len(predictions_df))]
#print(pred)
ind_col=pd.DataFrame(list1,columns=['index'])
#print(ind_col)
predictions_df.insert(1, "pred_index", ind_col['index'])
#pred.index=pred[pred_index]
predictions_df.set_index("pred_index",inplace=True)


#X=X.reset_index()
#print(predictions_df)
#print(new_train)
ax.plot(new_train, label='original', color='blue')
ax.plot(predictions_df, label='predictions', color='red')
#ax.axvline(x=len(new_train)+1,color='k', linestyle='--')
#ax.legend(loc='upper left')
plt.show()









