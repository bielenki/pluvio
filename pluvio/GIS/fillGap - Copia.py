import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix


method = "Mean"
os.chdir('c:\pluvio')

fileCSV = 'dadosPlu.csv'
fileSHP = 'GIS\estacoesPluCGS.shp'
indexData = 'data'
indexSHP = 'codigo'
dataPlu = pd.read_csv(fileCSV, index_col=indexData)
gagePlu = gpd.read_file(fileSHP)


gagePlu.set_index(indexSHP)
indexList = dataPlu.index.values.tolist()
srid = CRS(gagePlu.crs)
lonMean=0
if srid.coordinate_system.name == 'ellipsoidal':
    extent = gagePlu.total_bounds
    utm_crs_list = query_utm_crs_info( datum_name="WGS 84", area_of_interest=AreaOfInterest( west_lon_degree=extent[0], south_lat_degree=extent[1], east_lon_degree=extent[2], north_lat_degree=extent[3], ), )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    gagePlu = gagePlu.to_crs(utm_crs)

means = dataPlu.mean()
stds = dataPlu.std()
colNames=list(dataPlu.columns)
rows, columns = dataPlu.shape
matrixCorr = dataPlu.corr()
arrayCorr = matrixCorr.to_numpy()

matrixDist = gagePlu.geometry.apply(lambda g: ((gagePlu.distance(g))))
arrayDist = matrixDist.to_numpy()

invMatrixDist = 1/matrixDist

for col in range(columns):
    for row in range(rows):
        if col==row:
            invMatrixDist[col][row]=0


matrixIndices = np.multiply(matrixCorr,invMatrixDist)
arrayIndices = matrixIndices.to_numpy()
arrayData = dataPlu.to_numpy()
arrayDataP = np.copy(arrayData)

arrayPosition=[]
for col in range(columns):
    pMeans = means[col]
    pStds = stds[col]

    for row in range(rows):

        if np.isnan(arrayData[row,col]):
            arrayPIndicesCopy = np.empty(shape=columns)
            rowData = np.empty(shape=columns)
            arrayPMedia = np.empty(shape=columns)
            arrayPStds = np.empty(shape=columns)
            arrayPCorr = np.empty(shape=columns)
            arrayPIndices = np.empty(shape=columns)
            arrayPDist = np.empty(shape=columns)
            arraySortIndices = np.empty(shape=columns)
            arraySelecao = np.empty(shape=5)
            precX = 0

            for i in range(columns):
                if not np.isnan(arrayData[row, i]):
                    rowData[i] = arrayData[row, i]
                    arrayPMedia[i] = means[i]
                    arrayPStds[i] = stds[i]
                    arrayPCorr[i] = arrayCorr[col, i]
                    arrayPIndices[i] = arrayIndices[col, i]
                    arrayPDist[i] = arrayDist[col, i]

                else:
                    rowData[i] = 0
                    arrayPMedia[i] = 0
                    arrayPStds[i] = 0
                    arrayPCorr[i] = 0
                    arrayPIndices[i] = 0
                    arrayPDist[i] = 0

            arrayPIndicesCopy = np.copy(arrayPIndices)
            arraySortIndices = np.sort(arrayPIndicesCopy)
            Cont = 0
            for valor in arraySortIndices:
                if valor > 0 and Cont <4:
                    Cont = Cont + 1
                    arraySelecao[Cont] = valor


            arraySelecao = arraySortIndices[-5:]
            #print(arraySelecao)
            arrayPos=[]
            for item in arraySelecao:
                position = np.where(arrayPIndices == item)[0][0]

                if method == "Mean":
                    precX = precX + ((1 / 5) * ((pMeans / arrayPMedia[position]) * rowData[position]))
                if method == "Correlation":
                    precX = precX + ((pStds / 5) * (((rowData[position] - arrayPMedia[position]) / arrayPStds[position]) * arrayPCorr[position]))
                if method == "InvDist":
                    somaDist=0
                    for item2 in arraySelecao:
                        position2 = np.where(arrayPIndicesCopy == item2)[0][0]
                        somaDist = somaDist + arrayPDist[position2]
                    precX = precX + ((arrayPDist[position]/somaDist) * rowData[position])

            if method == "Mean":
                arrayDataP[row,col] = precX
            if method == "Correlation":
                arrayDataP[row,col] = precX + pMedia
            if method == "InvDist":
                arrayDataP[row,col] = precX

            del rowData
            del arrayPMedia
            del arrayPStds
            del arrayPCorr
            del arrayPIndices
            del arrayPIndicesCopy
            del arraySortIndices
            del arraySelecao





df = pd.DataFrame(arrayDataP, columns = colNames, index = indexList)
df.to_csv('preenchida.csv')


dfP = pd.DataFrame(arrayPosition)
dfP.to_csv('posicao.csv')

dfP2 = pd.DataFrame(matrixDist)
dfP2.to_csv('distancias.csv')

dfP3 = pd.DataFrame(matrixCorr)
dfP3.to_csv('correlacao.csv')






