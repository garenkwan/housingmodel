import json, os, shapefile, threading
import geopy.distance
import pandas as pd

from datetime import datetime
from numpy import nan
from svy21 import SVY21
from tqdm import tqdm

'''
Future work:
1. Check for duplicates where a nan became a value, update and drop duplicates
2. Does the data ever change? If yes, update with newer value --> do this only on the 'new' dataset to reduce compute
'''

#######################
#   Combining JSONs   #
#######################

def get_df(files):
    '''
    files: list of json from URA
    '''
    #date prep
    main = pd.DataFrame()
    list_of_jsons = []

    threads = []
    for file in files:
        process = threading.Thread(target=get_json, args=(file,list_of_jsons))
        process.start()
        threads.append(process)
    for process in threads:
        process.join()
    for file in list_of_jsons:
        main = pd.concat([main, file])
    main = main.reset_index(drop=True)
    
    #data cleaning
    main['cleanTenure'] = main.apply(lambda x: cleanTenure(x['tenure'], x['street']), axis=1)
    main['startYear'] = main.apply(lambda x: startYear(x['tenure'], x['street']), axis=1)
    main['Age'] = main['startYear'].apply(ageCalculator)
    main['remainingLease'] = main.apply(lambda x: remainingLease(x['cleanTenure'], x['startYear']),
                                        axis = 1)
    main['price'] = main['price'].astype('float')
    main['area'] = main['area'].astype('float')
    main['psm'] = main['price']/main['area']
    main['psf'] = main['psm']/10.764
    main['Date'] = main['contractDate'].apply(dateCleaner)
    main['sqft'] = main['area']*10.764
#     display(main)
    return main.drop_duplicates().reset_index(drop=True)

def get_json(file,list_of_jsons):
    print(datetime.now(), ':\t', 'Working on', file,'\n')
    df = pd.DataFrame()
    with open(file, 'r') as f:
        read_file = json.load(f)
        for entry in read_file['Result']:
            temp = pd.DataFrame(entry['transaction'])
            temp['street'] = entry['street']
            temp['project'] = entry['project']
            temp['marketSegment'] = entry['marketSegment']
            if 'x' in entry.keys():
                temp['x'] = entry['x']
            else:
                temp['x'] = nan
            if 'y' in entry.keys():
                temp['y'] = entry['y']
            else:
                temp['y'] = nan
            df = pd.concat([df, temp], sort=True)
    list_of_jsons.append(df)
    print(datetime.now(), ':\t', file,'completed.\n')
    return df

def get_latest_data(directory='./jsons/'):
    files = list(os.walk(directory))[0][2]
    max_date = 0
    for file in files:
        if '_' in file:
            date = file.split('_')[0]
            if date.isnumeric() and int(date)>max_date:
                max_date = int(date)
    files = [directory+file for file in files if str(max_date) in file and file.split('.')[-1] == 'json']
#     print(files)
    return files, max_date

#####################
#   Data Cleaning   #
#####################

def ageCalculator(year):
    if year == 1000:
        return 0
    else:
        return 2021-year

def cleanTenure(string, street):
    if string == 'Freehold':
        return 1000
    elif string == '99 years leasehold':
        return 99
    elif string == '110 Yrs From 01/11/2017':
        return 110
    elif string == 'NA':
        if street == 'JALAN NAUNG':
            return 999
        else:
            return 1000
    else:
        return int(string.split(' yrs lease commencing from ')[0])
    
def dateCleaner(date):
    date = str(date)
    year = str(int(date[-2:])+2000)
    if len(date) == 3:
        month = '0'+date[0]
    else:
        month = date[:2]
    return pd.to_datetime(month+'-01-'+year)

def preprocess(projectName, bins=11):
    '''
    uses main as df
    supply projectName
    '''
    global main
    df = main[main['project'] == projectName].copy()
    df['sqft'].hist(bins=bins)
    return df

def remainingLease(year, start):
    if year == 1000:
        return 1000
    else:
        return year-2021+start

def sizeBinning(df, listOfSizes=[829,1184,1500], feature='psf', debug=False):
    '''
    listOfSizes must be a list of length 3
    feature: 'price', 'psm', 'psf' or any other numeric in the df
    '''

    def areaBin(area):
        if area < listOfSizes[0]:
            return 'Small'
        elif area < listOfSizes[1]:
            return 'Medium'
        elif area < listOfSizes[2]:
            return 'Large'
        else:
            return 'Very Large'
        
    df['areaBin'] = df['area'].apply(areaBin)
    floors = df['floorRange'].unique().tolist()
    floors.sort()
    floors = floors[::-1]
    areasize = ['Small', 'Medium', 'Large', 'Very Large']
    if debug:
        print(floors)
        print(areasize)
    
    if len(floors) < 2 or len(areasize) < 2:
        if len(floors) == len(areasize):
            df.plot.scatter(x='Date', y=feature, title=floors[0]+' '+areasize[0])
        else:
            fig, ax = plt.subplots(len(floors),len(areasize),figsize=[4*len(areasize),4*len(floors)])
            if len(floors) < 2:
                for area in areasize:
                    df[df['areaBin']==area].plot\
                    .scatter(x='Date', y=feature, title=floors[0]+' '+area, 
                             ax=ax[areasize.index(area)])
            else:
                for floor in floors:
                    df[df['floorRange']==floor].plot\
                    .scatter(x='Date', y=feature, title=floor+' '+areasize[0], 
                             ax=ax[floors.index(floor)])
    else:
        fig, ax = plt.subplots(len(floors),len(areasize),figsize=[4*len(areasize),4*len(floors)])
        for floor in floors:
            for area in areasize:
                df[(df['floorRange']==floor) & (df['areaBin']==area)]\
                .plot.scatter(x='Date', y=feature, title=floor+' '+area, ax=ax[floors.index(floor)][areasize.index(area)])
                
def startYear(string, street):
    if string == 'Freehold':
        return 1000
    elif string == '99 years leasehold':
        return 2020
    elif string == '110 Yrs From 01/11/2017':
        return 2017
    elif string == 'NA':
        if street == 'JALAN NAUNG':
            return 1883
        else:
            return 1000
    else:
        return int(string.split(' yrs lease commencing from ')[1])
                
                
########################
#   Merge and Update   #
########################

def coerce_dtypes(df):
    df['area'] = df['area'].round(1)
    df['nettPrice'] = df['area'].round(0)
    df['price'] = df['price'].round(0)
    df['x'] = df['x'].round(5)
    df['y'] = df['y'].round(5)
    df['psm'] = df['psm'].round(2)
    df['psf'] = df['psf'].round(2)
    df['sqft'] = df['sqft'].round(0)
    return df

def convert_lat_long(main):
    '''
    Example for how to convert x and y into lat and long.
    Not used as most can be acheived in lookup table.
    '''
    convert = SVY21()
    latlon = []
#     print(main)
    for x,y in zip(main['x'], main['y']):
        if pd.notna(x) and pd.notna(y):
            latlon.append(convert.computeLatLon(x, y))
        else:
            latlon.append([x,y])
    latlon = pd.DataFrame(latlon, columns=['Lat', 'Lon'])
    main = pd.concat([main, latlon], axis=1)
    return main

def merge_data(new,previous='./PMI_Res_Transaction.csv'):
    '''
    1. Find the intersection of both datasets, keep index to preserve date ingested
    2. Get the data available in master but not in new
    3. Get the data available in new but not in master
    4. Concat and serve
    '''
    master = pd.read_csv(previous)
    master = coerce_dtypes(master)
    new = coerce_dtypes(new)
    cols = [x for x in list(master.columns) if x != 'date_ingested']
    colsd = cols+['date_ingested']
    
    #1 preserve index, intersection should take date_ingested from master
    master['master_index'] = master.index
    print('{}\nSize of master data:\t{} rows\nSize of new data:\t{} rows'\
          .format(len(cols), master.shape[0],new.shape[0]))
#     print(master.columns)
#     print(new.columns)
    intersection = pd.merge(new[cols], master[cols+['master_index']], how='inner',on=cols)
    intersection['date_ingested'] = master.loc[intersection['master_index'],'date_ingested'].to_list()
    intersection = intersection[colsd]
    print('Size of intersection:\t{} rows\nExpected number of new rows:\t{}'\
          .format(intersection.shape[0],new.shape[0]-intersection.shape[0]))
#     print('Intersection dates:\t', intersection['date_ingested'].unique())
    
    #2, 3 simply, if cannot be found in new, then 'n' isna, if cannot be found in master, then 'm' isna
    master['m']='m'
    new['n']='n'
    right = pd.merge(master[colsd+['m']], new[cols+['n']], how = 'left')
    base = right.loc[pd.isna(right['n']),colsd]
#     print('Base dates:\t', base['date_ingested'].unique())
    
    left = pd.merge(new[colsd+['n']], master[cols+['m']], how='left')
    addition = left.loc[pd.isna(left['m']),colsd]
    print('Added {} rows'.format(addition.shape[0]))
#     print('Addition dates:\t', addition['date_ingested'].unique())
    
    #4
    df = pd.concat([base,intersection,addition]).sort_values(['project', 'Date'])
    print('Size of new dataset:\t{} rows'.format(df.shape[0]))
    return df

def nearestStation(projectList, stationList):
    '''
    The table is not very large, so will be lazy and make one big table.
    '''
    results = []
    for project in tqdm(projectList['project']):
        distance = []
        lat, lon = projectList.loc[projectList['project']==project, ['Lat', 'Lon']].iloc[0]
        for idx in stationList.index:
            _,_,_,_,stnLat,stnLon = stationList.loc[idx,:]
#             print(lat, lon, stnLat, stnLon)
            distance.append(geopy.distance.distance((stnLat, stnLon),
                                                    (lat, lon)).km)
        min_dist = min(distance)
        results.append(stationList.loc[distance.index(min_dist),:].tolist() + [min_dist])
    
    colNames = ['STN_NAME', 'STN_NO', 'STN_SVY21_X', 'STN_SVY21_Y', 'STN_Lat', 'STN_Lon', 'Project_to_STN_dist']
    results = pd.DataFrame(results, columns=colNames)
    results = pd.concat([projectList, results],axis=1)
    return results

def updateProjectLocations(projectList):
    '''
    Start with the baseline projecList.csv, update with any new projects found in projectList
    projectList is a df with [proj, Lat, Lon]
    '''
    projPath = './ref_data/projectLocations.csv'
    stnPath = './ref_data/stations.csv'
    
    projects = pd.read_csv(projPath)
    currentProjects = list(projects['project'])
    newProjects = [proj for proj in projectList['project'].unique() if proj not in currentProjects]
#     print('New Projects', newProjects, len(newProjects))
    if len(newProjects) != 0:
        print('Updating with following projects:\n{}'.format(newProjects))
        stations = pd.read_csv(stnPath)
        projectsToUpdate = projectList.loc[(projectList['project'].isin(newProjects))\
                                           & (pd.notna(projectList['x']))\
                                           & (pd.notna(projectList['y'])),:]\
        .reset_index(drop=True)
        newProjectsDF = convert_lat_long(projectsToUpdate)
        updateProjects = nearestStation(newProjectsDF,stations)
        projects = pd.concat([projects,updateProjects])
#         print('Columns Names', projects.columns)
        colNames = [x for x in projects.columns if (x != 'x') and (x != 'y')]
        projects[colNames].to_csv(projPath, index = False)
    else:
        print('No projects to update for.')

        
###########
#   RUN   #
###########

                
if __name__ == '__main__':
    files, date = get_latest_data()
    new = get_df(files)
    newFileName = str(date) + '_PMI_Res_Transaction.csv'
    new['date_ingested']=str(date)
    new.to_csv(newFileName, index=False)
    print('New data set saved to:\t{}'.format(newFileName))

    #to coerce similar data types between master and new, just read in new
    new = pd.read_csv('./'+newFileName)
    updatedDF = merge_data(new)
    os.rename('./PMI_Res_Transaction.csv','./ARCHIVE_BEFORE_'+str(date)+'_UPDATE.csv')
    updatedDF.to_csv('./PMI_Res_Transaction.csv', index=False)
    
    # update train stations
    # have files called stations.csv, projectLocations.csv
    updateProjectLocations(updatedDF[['project', 'x', 'y']])
    print('Project locations with nearest MRT updated.')
    projectLocations = pd.read_csv('./ref_data/projectLocations.csv')
    colNames = ['project', 'STN_NAME', 'STN_NO', 'STN_SVY21_X', 'STN_SVY21_Y', 'STN_Lat', 'STN_Lon', 'Project_to_STN_dist']
    stnsDF = pd.merge(updatedDF, projectLocations, on='project', how = 'left')
    stnsPath = str(date)+'_PMI_Res_Transaction_w_STN.csv'
    #this is to snapshot the data
    stnsDF.to_csv(stnsPath, index=False)
    #this is to update the file to latest
    stnsDF.to_csv('PMI_Res_Transaction_w_STN.csv', index=False)
    print('Joined with nearest MRT station and saved to {}.'.format(stnsPath))