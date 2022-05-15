import urllib
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
from flask import Flask, render_template, request, url_for,redirect
#from classifier import myfunc
import firebase_admin
from firebase_admin import credentials, firestore
from firebase import firebase
import math
import pyrebase
import pandas as pd
from pandas.api.types import CategoricalDtype
from statsmodels.miscmodels.ordinal_model import OrderedModel


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
firestore_db = firestore.client()

firebase=firebase.FirebaseApplication("https://svhms-user-details-default-rtdb.firebaseio.com/",None)
firebaseConfig = {
    'apiKey': "AIzaSyD3-UPt-sP6wPtyDN2TLPHKy5qua-J3_04",
    'authDomain': "svhms-user-details.firebaseapp.com",
    'databaseURL': "https://svhms-user-details-default-rtdb.firebaseio.com",
    'projectId': "svhms-user-details",
    'storageBucket': "svhms-user-details.appspot.com",
    'messagingSenderId': "173656186274",
    'appId': "1:173656186274:web:e944d76486a9017a0026fb",
    'measurementId': "G-XQCNVHNW2W"
  }
firebase=pyrebase.initialize_app(firebaseConfig)
auth=firebase.auth()
storage=firebase.storage()
doc_ref=""
sub_ref=""
cloudfilename=""


app = Flask(__name__)
userDetails={}
def get_df():
    global cloudfilename
    url = storage.child(cloudfilename).get_url(None)
    print(url)
    try:
        df = pd.read_csv(url, index_col=False)
        #print("returning df")
        return df
    except Exception as e:
        print(e)
    #print("returning 0")
    return 0

def get_dashboard_values():
    df = get_df()
    #print("get dash val :",type(df))
    if isinstance(df, pd.core.frame.DataFrame):
        #print("inside get dashboard values func")
        send_list_values = []
        runtime=math.ceil(max(df['ENGINE_RUN_TINE ()']) / 60)
        units="Minutes"
        if runtime>60:
            runtime = math.ceil(runtime / 60)
            units='Hours'
        send_list_values.append(runtime)
        send_list_values.append(round(max(df['ENGINE_RPM ()']), 2))
        send_list_values.append(round(df['ENGINE_RPM ()'].mean(), 2))
        send_list_values.append(df['ENGINE_RPM ()'].tolist()) #engine rpm
        send_list_values.append(df['ENGINE_RUN_TINE ()'].tolist()) #engine run time
        send_list_values.append(df['VEHICLE_SPEED ()'].tolist()) #engine vehicle speed
        #send_list_values.append(math.ceil(df['COOLANT_TEMPERATURE ()'].mean())) #coolant temp
        #send_list_values.append(math.ceil(df['INTAKE_AIR_TEMP ()'].mean())) #intake temp
        send_list_values.append(round(max(df['COOLANT_TEMPERATURE ()']), 2))
        send_list_values.append(round(max(df['INTAKE_AIR_TEMP ()']), 2))

        send_list_values.append(round(df['VEHICLE_SPEED ()'].mean(), 2)) #vehicle speed
        send_list_values.append(units)

        return send_list_values
    else:
        #print("returning 0")
        return df


df = pd.read_csv('train.csv')


def preprocessing():
    global df
    df.head()
    df['INTAKE_AIR_TEMP']=pd.to_numeric(df['INTAKE_AIR_TEMP'],errors='coerce')
    df = df.dropna()
    print("Preprocessing done")

def modelling():
    classify_type = CategoricalDtype(categories=['Good', 'Moderate', 'Bad'], ordered=True)
    df['RESULT'] = df['RESULT'].astype(classify_type)
    mod_prob = OrderedModel(df['RESULT'],
                            df[['INTAKE_AIR_TEMP', 'COOLANT_TEMPERATURE', 'ENGINE_RPM']],
                            distr='logit')
    res_log = mod_prob.fit(method='bfgs')
    print("Modelling Done")
    return res_log

def prediction(df):
    global res_log
    my_predicted = res_log.model.predict(res_log.params,
                                         exog=df[['INTAKE_AIR_TEMP', 'COOLANT_TEMPERATURE', 'ENGINE_RPM']])
    list1=my_predicted.tolist()
    return list1

def myfunc(intake_air,coolant,engine_rpm):

    data = {'INTAKE_AIR_TEMP': [intake_air],
            'COOLANT_TEMPERATURE': [coolant],
            'ENGINE_RPM': [engine_rpm]
            }
    df1 = pd.DataFrame(data)
    res_list=prediction(df1)
    elem = max(res_list[0])
    # index
    res = res_list[0].index(elem)
    return res

preprocessing()
res_log = modelling()


def get_health_values_from_file():
    df=get_df()
    send_list_values=[]
    send_list_values.append(round(df['ENGINE_RPM ()'].mean(), 2))
    send_list_values.append(round(df['INTAKE_AIR_TEMP ()'].mean(), 2))  # intake temp
    send_list_values.append(round(df['COOLANT_TEMPERATURE ()'].mean(), 2))  # coolant temp
    return send_list_values


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/choose_login')
def choose_login():
    return render_template('login.html',msg="")

@app.route('/login', methods=['POST', 'GET'])
def login():
    global userDetails,doc_ref,cloudfilename
    if request.method == 'POST' and len(dict(request.form)) > 0:
        userdata = dict(request.form)
        email = userdata["email"]
        password = userdata["password"]
        try:
            login = auth.sign_in_with_email_and_password(email, password)
            userID=login['localId']
            doc_ref=firestore_db.collection(u'users').document(userID)
            doc_ref_get = firestore_db.collection(u'users').document(userID).get()
            userDetails=doc_ref_get.to_dict()
            #print(userDetails)
            cloudfilename=userDetails['vehicle_id']+".csv"
            #print(cloudfilename)
            send_list_values=get_dashboard_values()
            #print("came back to login & returned val:",send_list_values)
            if send_list_values==0:
                return render_template("nodatapage.html",msg="NO such files uploaded from the car",userDetails=userDetails)
            else:
                return render_template("userhomepage.html",pred_index=0,actual_dist=[],pred_dist=[],coolant_temp=send_list_values[6],flag="",avg_speed=send_list_values[8],intake_air_temp=send_list_values[7],userDetails=userDetails,list_values=send_list_values,engine_rpm=send_list_values[3],engine_runtime=send_list_values[4],vehicle_speed=send_list_values[5],units=send_list_values[-1])

        except:
            msg="Invalid email or password"
            return render_template('login.html',msg=msg)


@app.route('/logout', methods=['POST', 'GET'])
def logout():
    auth.current_user = None
    return render_template('index.html')

@app.route('/choose_register')
def choose_register():
    return render_template('register.html',msg="")

@app.route('/register', methods=['POST', 'GET'])
def register():
    global doc_ref,sub_ref
    if request.method == 'POST' and len(dict(request.form)) > 0:
        userID=""
        userdata = dict(request.form)
        email = userdata["email"]
        car = userdata["car"]
        vehicle_id = userdata["vehicle_id"]
        password = userdata["password"]
        #new_data = {"email": email, "car": car, "model": model, "password": password}
        #print(new_data)
        #firebase.post("/users", new_data)
        try:
            user = auth.create_user_with_email_and_password(email, password)
            userID = user['localId']
            #print(userID)
            doc_ref=firestore_db.collection(u'users').document(userID)
            #print(doc_ref)
            doc_ref_set=doc_ref.set({'email': email, 'car': car,'vehicle_id':vehicle_id,'password':password})
            #print(doc_ref,doc_ref_set)
            sub_ref=doc_ref.collection(u'livedata').document(u'samplelive').set({'IntakeAirTemp': "65",'EngineRPM':'750','CoolantTemperature':"80"})
            #print(sub_ref)
            return render_template("index.html")
        except:
            msg="Email already exists"
            return render_template('register.html',msg=msg)

@app.route('/checkhealth')
def checkhealth():
    global userDetails
    print("inside check health function")
    list1=get_health_values_from_file()
    res=myfunc(list1[1],list1[2],list1[0])
    if res==0:
        res_word="GOOD"
        msg="Happy Travel"
    elif res==1:
        res_word="MODERATE"
        msg="Sooner Service required"
    else:
        res_word="BAD"
        msg="Immediate Service Required"
    send_list_values = get_dashboard_values()
    return render_template("userhomepage.html",pred_index=0,actual_dist=[],pred_dist=[], coolant_temp=send_list_values[6], avg_speed=send_list_values[8],
                           intake_air_temp=send_list_values[7], userDetails=userDetails, list_values=send_list_values,
                           engine_rpm=send_list_values[3], engine_runtime=send_list_values[4],flag="",
                           vehicle_speed=send_list_values[5],units=send_list_values[-1],res=res,res_word=res_word,type='Health',msg=msg)


@app.route('/livecheckhealth')
def livecheckhealth():
    global userDetails
    print("inside check health function")
    sub_ref_get = doc_ref.collection(u'livedata').document(u'samplelive').get()
    dict1 = sub_ref_get.to_dict()
    print(dict1)
    res=myfunc(float(dict1['IntakeAirTemp']),float(dict1['CoolantTemperature']),float(dict1['EngineRPM']))
    if res==0:
        res_word="GOOD"
        msg='Happy Travel'
    elif res==1:
        res_word="MODERATE"
        msg="Sooner Service required"
    else:
        res_word="BAD"
        msg="Immediate Service Required"
    send_list_values = get_dashboard_values()
    return render_template("userhomepage.html", pred_index=0,actual_dist=[],pred_dist=[],coolant_temp=send_list_values[6], avg_speed=send_list_values[8],
                           intake_air_temp=send_list_values[7], userDetails=userDetails, list_values=send_list_values,
                           engine_rpm=send_list_values[3], engine_runtime=send_list_values[4],
                           vehicle_speed=send_list_values[5],units=send_list_values[-1],res=res,flag="",res_word=res_word,type='Live Health',msg=msg)

@app.route('/knowmore')
def knowmore():
    mydata_ref1 = pd.read_csv('dataset.csv', parse_dates=True)
    flag="1"
    # print(mydata_ref1.head())
    p = range(1, 21)
    d = range(1, 4)
    q = range(0, 4)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    # print(pdq)
    # print(len(pdq))
    pdq = (19, 3, 2)  # pdq=find_best_fit()
    # p,d,q=pdq[0],pdq[1],pdq[2]
    # print(type(pdq))

    # fit model
    x = mydata_ref1['distance travelled with remaining fuel']
    model = ARIMA(x, order=pdq)
    model_fit = model.fit()
    model_fit0 = model_fit

    # split data into train and test and forecast test
    X = mydata_ref1['distance travelled with remaining fuel']
    size = int(len(X) * 0.6)
    train = mydata_ref1['distance travelled with remaining fuel'].iloc[:size]
    test = mydata_ref1['distance travelled with remaining fuel'].iloc[size:]
    # print(test)
    # Fit ARIMA(19,3,2) on train data and find parameters
    model = ARIMA(train, order=(19, 3, 2))
    model_fit = model.fit(start_params=model_fit0.params)
    # predict train data
    print('Forecasting Training Data')
    pred_train = model_fit.predict(start=1, end=size)
    # print(pred_train)
    MSE_train = mean_squared_error(train, pred_train)
    RMSE_train = sqrt(MSE_train)
    # RMSPE_train = RMSPE(train,pred_train)

    print('train RMSE: %.5f' % RMSE_train)
    # print('train RMSPE: %.5f' % RMSPE_train)
    pred_train = pred_train.tolist()
    # print("pred_train",pred_train)
    print('Forecasting Test Data')
    # predict test data
    history = [x for x in train]
    pred_test = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(19, 3, 2))
        model_fit = model.fit(start_params=model_fit.params)
        output = model_fit.forecast()
        yhat = output[0]
        yhat = yhat.tolist()
        pred_test.append(yhat)
        obs = test[t + size]
        history.append(obs)
    # print('predicted=%f, expected=%f, difference=%f' % (yhat, obs,yhat-obs))
    MSE_test = mean_squared_error(test, pred_test)
    RMSE_test = sqrt(MSE_test)
    # RMSPE_test = RMSPE(test,pred_test)
    print('test RMSE: %.5f' % RMSE_test)
    # print('Test RMSPE: %.5f' % RMSPE_test)
    predictions = np.concatenate((pred_train, pred_test), axis=0)
    # print(pred_test)
    # print(len(pred_test))

    # line plot of observed vs predicted

    fig, ax = plt.subplots(1)
    predictions_df = pd.DataFrame(predictions, columns=['predicted'])
    new_train = train
    new_train = new_train.reset_index()
    list1 = [x for x in range(len(new_train), len(new_train) + len(predictions_df))]
    pred_index=predictions_df.index.values.tolist()
    actual_index=new_train.index.values.tolist()
    actual_dist=new_train['distance travelled with remaining fuel'].tolist()
    pred_dist=predictions_df['predicted'].tolist()
    send_list_values = get_dashboard_values()
    return render_template("userhomepage.html", coolant_temp=send_list_values[6], avg_speed=send_list_values[8],
                           intake_air_temp=send_list_values[7], userDetails=userDetails, list_values=send_list_values,
                           engine_rpm=send_list_values[3], engine_runtime=send_list_values[4],
                           vehicle_speed=send_list_values[5],units=send_list_values[-1],flag=flag,actual_index=actual_index,pred_index=pred_index,actual_dist=actual_dist,pred_dist=pred_dist)




if __name__ == '__main__':
    app.run(debug=True)