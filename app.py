import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

@st.cache(persist=True)
def get_estimate(projName, size, street=None):
    '''
    projName: supply project name
    size: size of unit in sqft
    street: used only if projName == 'LANDED HOUSING DEVELOPMENT' or 'RESIDENTIAL APARTMENTS'
            used to specify the street that the house is on
    returns chart of transactions (past 5 years) and estimates of transaction price for unit under consideration
    '''
    global df
    projName = projName.upper()
    if projName == 'LANDED HOUSING DEVELOPMENT' or projName == 'RESIDENTIAL APARTMENTS':
        temp = df.loc[(df['street']==street), ['project','Date','psf','price', 'sqft']].sort_values('Date')
    else:
        temp = df.loc[(df['project']==projName), ['project','Date','psf','price', 'sqft']].sort_values('Date')
    if temp.shape[0] == 0:
        text = 'Error, no data in dataframe.'
        return text, '_', '_', '_'
    temp['date'] = temp['Date']-temp['Date'].min()
    temp['date'] = temp['date'].apply(lambda x: x.days+1)
    temp['street'] = street

    lr = lr_time(temp)
    now, est_psf, r_square, rmse, min_psf, max_psf = metrics(lr, temp)

    text = '{}\n\
        Estimated psf for {:.0f}sqft sized unit = ${:.0f} \n\
        Total price = ${:.0f}\n\
        \tLinear Regression: R-square ({:.4f}), RMSE (${:.1f})\n\
        Price at min psf (${:.0f}): ${:.0f}\n\
        Price at max psf (${:.0f}): ${:.0f}'\
              .format(projName, size, est_psf, size*est_psf,
                      r_square,rmse,
                      min_psf, min_psf*size, max_psf, max_psf*size)
    temp['Date']=temp['Date'].dt.date
    return text, temp

@st.cache(persist=True)
def extended_estimate(projName, size, extension='street', tuning=0.5, self_prop=True):
    '''
    projName: supply project name
    size: size of unit in sqft
    extension: which feature to use to extend the search on
    tuning: factor to tune size of units under consideration
    self_prop: include all self properties, set to false if there are \
    odd sized units in the same property that need to be excluded \
    should there be a need to limit to units close to target unit size

    returns chart of transactions (past 5 years) and estimates of transaction price for unit under consideration
    '''
    global df
    features = ['Date','project','psf','price','sqft']
    projName = projName.upper()
    extendedBy = df.loc[df['project']==projName, extension].unique()[0]
    temp = df.loc[(df[extension]==extendedBy), features]
    #do some preprocessing to drop outliers
    nbrSTDev = 1
    stdev = temp['psf'].std()
    tempMean = temp.loc[temp['project']==projName, 'psf'].mean()
    projMax = temp.loc[temp['project']==projName, 'psf'].max()
    projMin = temp.loc[temp['project']==projName, 'psf'].min()
    while ((tempMean+stdev*nbrSTDev < projMax) or (tempMean-stdev*nbrSTDev>projMin)) and (nbrSTDev<3):
        nbrSTDev += 1
#     print(nbrSTDev)
    selectionMin = tempMean - nbrSTDev*stdev
    selectionMax = tempMean + nbrSTDev*stdev
    temp = df.loc[(df[extension]==extendedBy) & (df['psf']<=selectionMax) & (df['psf']>=selectionMin)\
                  & (df['sqft'] < size*(1+tuning)) & (df['sqft'] > size*(1-tuning)),
                  features].drop_duplicates().sort_values('Date').copy()
    if self_prop and tuning==0.5:
        incase = df.loc[df['project']== projName, features].copy()
        temp = pd.concat([temp, incase], sort=True).drop_duplicates().sort_values('Date')

    temp['date'] = temp['Date']-temp['Date'].min()
    temp['date'] = temp['date'].apply(lambda x: x.days+1)

    lr = lr_time(temp)
    now, est_psf, r_square, rmse, min_psf, max_psf = metrics(lr, temp)

    text = '{}\n\
        Estimated psf for {:.0f}sqft sized unit = ${:.0f}\n\
        Total price = ${:.0f}\n\
        \tLinear Regression: R-square ({:.4f}), RMSE (${:.1f})\n\
        Price at min psf (${:.0f}): ${:.0f}\n\
        Price at max psf (${:.0f}): ${:.0f}'\
          .format(projName, size, est_psf, size*est_psf,
                  r_square,rmse,
                  min_psf, min_psf*size, max_psf, max_psf*size)
    temp['Date']=temp['Date'].dt.date
    return text, temp

def drop_street_extension(streetName):
    '''
    Drop the prefix or suffix of a street name to help with extended_estimate
    Prefix dropped: LORONG
    Suffix dropped: WALK, CLOSE, ROAD, AVENUE, DRIVE
    E.g. the following will all become 'HOW SUN'
    'LORONG HOW SUN', 'HOW SUN WALK', 'HOW SUN CLOSE', 'HOW SUN ROAD',
       'HOW SUN AVENUE', 'HOW SUN DRIVE'
    '''
    if streetName.startswith('LORONG '):
        return streetName[7:]
    for suffix in ['WALK', 'CLOSE', 'ROAD', 'AVENUE', 'DRIVE', 'VALE', 'STREET', 'LINK', 'LANE', 'CRESCENT']:
        if streetName.endswith(suffix):
            length = len(suffix)+1
            return streetName[:-length]
    return streetName

@st.cache(persist=True)
def loadData(path):
    df = pd.read_csv(path)
    df['Date'] = df['Date'].apply(pd.to_datetime)
    df['Date.Year']=df['Date'].apply(lambda x: x.year)
    df['sqft'] = df['sqft'].apply(int)
    df['drop_street_extension'] = df['street'].apply(lambda x: drop_street_extension(x))
    return df

def lr_time(temp):
    #lr of data taking time and sqft (only)
    #fix to limit data to past 3 years
    if temp['date'].max() > 1095:
        minDate = temp['date'].max()-1095
        lr = LinearRegression().fit(np.array(\
        temp.loc[(temp['date'] > minDate), ['date', 'sqft']]),
        temp.loc[(temp['date'] > minDate), 'psf'])
    else:
        lr = LinearRegression().fit(np.array(temp[['date', 'sqft']]), temp['psf'])

    return lr

def metrics(lr, temp):
<<<<<<< Updated upstream
    global size
=======
>>>>>>> Stashed changes
    now = (datetime.now()-temp['Date'].min()).days
    est_psf = lr.predict(np.array([now, size]).reshape(1,-1))[0]
    r_square = lr.score(temp[['date', 'sqft']], temp['psf'])
    rmse=np.sqrt(mse(lr.predict(temp[['date', 'sqft']]), temp['psf']))
    min_psf = temp['psf'].min()
    max_psf = temp['psf'].max()

    return now, est_psf, r_square, rmse, min_psf, max_psf

path = './data/PMI_Res_Transaction_w_STN.csv'
df = loadData(path)
streetSelector = {'Same street':'street', 'Whole area':'drop_street_extension'}
allprojects = df['project'].unique()

# All sidebar controls
st.sidebar.title('Property Options')
project = st.sidebar.text_input('Project name', '')
project = project.upper()
if project not in df['project'] and project != '':
    projects = [x[0] for x in process.extract(project, allprojects, limit=10) if x[1]>80]
    projects = projects[:5] if len(projects) > 5 else projects
    project = st.sidebar.radio('Select project', projects, 0)
size = st.sidebar.text_input('Property size', '')
size = int(size) if size != '' else size
extended = st.sidebar.checkbox('Include other projects on same street', value=False)
if extended:
    street = st.sidebar.radio('',['Same street', 'Whole area'])
    streetSelect = streetSelector[street]
    tuning = st.sidebar.slider('Tuning Factor for property size', 0.1, 0.5, 0.3, 0.05)

# Main display
st.title('Quick price estimate')
st.subheader('Data as at 26-Oct-21')

if project == '' or size == '':
    if (project == '' and size == '') or project == '':
        text = 'Please input name of property'
    else:
        text = 'Please input size of property'
    smallDF = pd.DataFrame([])
else:
    if not extended:
        text, smallDF = get_estimate(project, size)
    else:
        text, smallDF = extended_estimate(project, size, streetSelect, tuning)
st.subheader(text)

if project != '' and size != '':
    if not extended:
        fig = px.scatter(smallDF, x='Date', y='price', color='psf', size='sqft')
        st.plotly_chart(fig)
    else:
        fig = px.scatter(smallDF, x='Date', y='price', color='project', size='sqft')
        st.plotly_chart(fig)

    st.write(smallDF[['Date','project', 'psf', 'price', 'sqft']].reset_index(drop=True))
