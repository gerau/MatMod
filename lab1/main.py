import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



table = pandas.read_csv("dataset.csv")
print(table)
table = table.drop('value_text', axis = 1)


print(table["phenomenon"].unique())

feature_df = table.groupby(['phenomenon', 'logged_at'], as_index=False).aggregate('mean')
tableall = feature_df.pivot_table(index=['logged_at'], columns='phenomenon', values=['value',])
print(tableall)
tableall.columns = [col[1] if col[1]!='' else col[0] for col in tableall.columns.values]

tableall = tableall.dropna()
tableall.reset_index(inplace=True)
tableall['logged_at'] = pandas.to_datetime(tableall['logged_at'])
tableall["hour"] = tableall['logged_at'].dt.hour.astype('float16') + tableall['logged_at'].dt.minute.astype('float16')/60 + tableall['logged_at'].dt.second.astype("float16")/3600
for column in tableall.columns:
    if(column == "logged_at"):
        continue
    tableall[column] = pandas.to_numeric(tableall[column], errors='coerce')

tableall_short = tableall.sample(n = 2000)

print(tableall)




fig, axs = plt.subplots(ncols=3, nrows=4)


sns.regplot(x = tableall_short['pm1'], y = tableall_short['pm25'], ax = axs[0,1])
sns.regplot(x = tableall_short['pm10'], y = tableall_short['pm25'], ax = axs[0,2])
sns.regplot(x = tableall_short['hour'], y = tableall_short['temperature'], ax = axs[1,0])
sns.regplot(x = tableall_short['pm1'], y = tableall_short['pm10'], ax = axs[0,0])
sns.regplot(x = tableall_short['hour'], y = tableall_short['humidity'], ax = axs[1,1])
sns.regplot(x = tableall_short['humidity'], y = tableall_short['temperature'], ax = axs[1,2])
sns.regplot(x = tableall_short['pm1'], y = tableall_short['temperature'], ax = axs[2,0])
sns.regplot(x = tableall_short['pressure_pa'], y = tableall_short['hour'], ax = axs[2,1])
sns.regplot(x = tableall_short['pm25'], y = tableall_short['humidity'], ax = axs[2,2])
sns.regplot(x = tableall_short['hour'], y = tableall_short['pm1'], ax = axs[3,0],color= "r")
sns.regplot(x = tableall_short['hour'], y = tableall_short['pm10'], ax = axs[3,1])
sns.regplot(x = tableall_short['hour'], y = tableall_short['pm25'], ax = axs[3,2])
plt.show()
fig, axs = plt.subplots(ncols=3, nrows=4)
sns.regplot(x = tableall_short['no2_ppb'], y = tableall_short['pm25'], ax = axs[0,1])
sns.regplot(x = tableall_short['no2_ppb'], y = tableall_short['o3_ppb'], ax = axs[0,2])
sns.regplot(x = tableall_short['hour'], y = tableall_short['o3_ppb'], ax = axs[1,0])
sns.regplot(x = tableall_short['humidity'], y = tableall_short['no2_ug'], ax = axs[0,0])
sns.regplot(x = tableall_short['temperature'], y = tableall_short['o3_ug'], ax = axs[1,1])
sns.regplot(x = tableall_short['pressure_pa'], y = tableall_short['no2_ug'], ax = axs[1,2])
sns.regplot(x = tableall_short['o3_ppb'], y = tableall_short['temperature'], ax = axs[2,0])
sns.regplot(x = tableall_short['pressure_pa'], y = tableall_short['o3_ug'], ax = axs[2,1])
sns.regplot(x = tableall_short['pm25'], y = tableall_short['humidity'], ax = axs[2,2])
sns.regplot(x = tableall_short['hour'], y = tableall_short['no2_ppb'], ax = axs[3,0])
sns.regplot(x = tableall_short['hour'], y = tableall_short['o3_ug'], ax = axs[3,1])
sns.regplot(x = tableall_short['hour'], y = tableall_short['no2_ppb'], ax = axs[3,2])
plt.show()



model = LinearRegression()
columns = tableall.columns
columns = columns.delete(0)
columnstemp = columns


for i in columns:
    x = tableall[i].to_numpy().reshape(-1,1)
    for j in columnstemp:
        if(i == j): continue
        y = tableall[j].to_numpy().reshape(-1,1)
        m = model.fit(x,y)
        print(f"{round(m.score(x,y),3)} - determination score for {i} and {j}; slope: {m.coef_}; intercept: {m.intercept_}" )
    columnstemp = columnstemp.delete(0)
   
 
plt.show()
model = LinearRegression()
x = pandas.DataFrame(tableall["no2_ppb"])
y = pandas.DataFrame(tableall["hour"])
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_test["no2_ppb_pred"] = y_pred
y_test['no2_ppb'] = x_test
print(y_test)
print(f"r2_score  ", r2_score(x_test, y_pred))
print(f"RMSE  ", mean_squared_error(x_test, y_pred, squared=True), '\n\n')



model = LinearRegression()
x = pandas.DataFrame(tableall["o3_ppb"])
y = pandas.DataFrame(tableall["o3_ug"])
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_test["o3_ppb"] = y_pred
y_test['o3_ppb_pred'] = x_test
print(y_test)
print(f"r2_score  ", r2_score(x_test, y_pred))
print(f"RMSE  ", mean_squared_error(x_test, y_pred, squared=True), '\n\n')


model = LinearRegression()
x = pandas.DataFrame(tableall["no2_ppb"])
y = pandas.DataFrame(tableall["temperature"])
x_train, x_test, y_train, y_test = train_test_split(x, y)
model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_test["no2_ppb"] = y_pred
y_test['no2_ppb_pred'] = x_test
print(y_test)
print(f"r2_score  ", r2_score(x_test, y_pred))
print(f"RMSE  ", mean_squared_error(x_test, y_pred, squared=True), '\n\n')