from MachineLearning_Utils.Data_Processing_Utils import *
from MachineLearning_Utils.Modeler import *
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, HuberRegressor, LassoCV, Lasso, ElasticNet, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

train_data = "C:/Users/Juan.Zarco/Documents/Random Datasets/Housing Prices/train.csv"
test = "C:/Users/Juan.Zarco/Documents/Random Datasets/Housing Prices/test.csv"


eda = EDA.load_data_from_file(fulldata_filepath=train_data, index_col='Id')

print(eda.describe_data())

#eda.scatter_plot('LotArea', 'SalePrice', 'LotArea', 'SalePrice', 'Scatter Plot')

#eda.corr_plot()

processor = DataProcessor.load_train_test(train_data,test,'SalePrice',index_col='Id')

_dict = processor.get_datasets()

print(processor)

pipe = {'impute':["mean"],
        'label_encoding':['one_hot',3],
        'label_encoding':['sum']}

data = processor.pipeline(steps=pipe)

#processor.label_encoding('one_hot',cardinality=3)
#processor.label_encoding('sum')

print(processor)

vif = processor.vif()

high_vif = vif[vif>5]

print(high_vif)

data = processor.get_datasets()

X_train = data['X_data']['X_train']
Y_train = data['Y_data']['Y_train']


X_val = data['X_data']['X_val']
Y_val = data['Y_data']['Y_val']

X_test = data['X_data']['X_test']

idx = X_train[X_train['LotArea'] > 100000].index
print(idx)
X_train.drop(idx, inplace=True)
Y_train.drop(idx, inplace=True)

Y_train_log = pd.Series(np.log(Y_train),index=Y_train.index)

lr = LinearRegression()
enet = ElasticNetCV(l1_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],n_alphas=200,alphas=[0.1,0.01,0.001,0.0001], max_iter=30000,cv=3)
ridge = RidgeCV(alphas=[0.1,0.35,0.5,0.75,1,5,10],cv=3)
huber = HuberRegressor(epsilon=2,max_iter=15000,alpha=0.001)
lasso = LassoCV(n_alphas=200,cv=3, max_iter=30000)
svr = LinearSVR(C=.85,loss='squared_epsilon_insensitive', tol=1e-5)

mse = make_scorer(mean_squared_error)

params = {'l1_ratio':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],  'alpha':(05.,0.1,0.05,0.01,0.005,0.001), 'max_iter':[3000,4000,5000]}
enet_grid = GridSearchCV(ElasticNet(),params,scoring=mse,cv=3)

params = {'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]}
ridge_grid = GridSearchCV(Ridge(),params,scoring=mse,cv=3)

params = {'epsilon':[1.35,2,2.35,3,3.35,3.75,4,5],'alpha':[05.,0.1,0.05,0.01,0.005,0.001]}
huber_grid = GridSearchCV(huber,params,scoring=mse,cv=3)

params = {'alpha':[0.5,0.1,0.05,0.01,0.005,0.001], 'tol':[1e-2,1e-3,1e-4,1e-5],'max_iter':[3000,4000,5000]}
lasso_grid = GridSearchCV(Lasso(),params,scoring=mse,cv=3)

params = {'tol':[1e-2,1e-3,1e-4,1e-5], 'C':[0.5,0.75,1, 5,10]}
svr_grid = GridSearchCV(svr,params,scoring=mse,cv=3)

rfg = RandomForestRegressor(n_estimators=1000)
gboost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1)

my_models = SupervisedLearner(model_list=[lr, enet_grid, ridge_grid, huber_grid, lasso_grid, svr_grid, rfg, gboost])

#my_models.fit(X_train,Y_train_log)

#predictions = my_models.predict(X_val)

#for name, prediction in predictions.items():
#    print("\nName: ", name)
#    scores = my_models.score_models(Y_val.values,np.exp(prediction))
#    print("\n",scores)
    #print("\nRMSE : {0:,.2f} ; MSE : {0:,.2f} ; MAE : {0:,.2f}".format(RMSE(Y_val,np.exp(prediction)), MSE(Y_val,np.exp(prediction)), MAE(Y_val,np.exp(prediction))))

meta = MetaLearner(rfg,model_list=[lr, enet_grid, ridge_grid, huber_grid, lasso_grid, svr_grid, rfg, gboost])

meta.fit(X_train,Y_train_log)
meta.meta_fit(X_train,Y_train_log)

y_pred = meta.meta_predict(X_val)
y_pred = np.exp(y_pred)

predictions = meta.predict(X_val)

for name, prediction in predictions.items():
    print("\nName: ", name)
    scores = my_models.score_models(Y_val.values,np.exp(prediction))
    print("\n",scores)
    #print("\nRMSE : {0:,.2f} ; MSE : {0:,.2f} ; MAE : {0:,.2f}".format(RMSE(Y_val,np.exp(prediction)), MSE(Y_val,np.exp(prediction)), MAE(Y_val,np.exp(prediction))))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

print("MSE: ",mean_squared_error(Y_val,y_pred))
print("RMSE: ",sqrt(mean_squared_error(Y_val,y_pred)))
print("MAE: ",mean_absolute_error(Y_val,y_pred))
print("R2: ",r2_score(Y_val,y_pred))
print("EVS: ",explained_variance_score(Y_val,y_pred))

y_test_predictions = meta.meta_predict(X_test)
y_test_predictions = np.exp(y_test_predictions)

Y_test = pd.Series(y_test_predictions,index=X_test.index).to_frame()
#Y_test.to_csv("Housing_Prices_Predictions.csv")