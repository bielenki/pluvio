import os
import numpy as np
import pandas as pd
import geopandas as gpd


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix


method = "Mean"
os.chdir('c:\pluvio')

fileCSV = 'dadosPlu.csv'
fileSHP = 'GIS\estacoesPlu.shp'
indexData = 'data'
indexSHP = 'codigo'
dataPlu = pd.read_csv(fileCSV, index_col=indexData)
gagePlu = gpd.read_file(fileSHP)


gagePlu.set_index(indexSHP)
indexList = dataPlu.index.values.tolist()


means = dataPlu.mean()
stds = dataPlu.std()
colNames=list(dataPlu.columns)
rows, columns = dataPlu.shape
matrixCorr = dataPlu.corr()
arrayCorr = matrixCorr.to_numpy()

matrixDist = gagePlu.geometry.apply(lambda g: gagePlu.distance(g))

#matrixDistN = normalize_2d(matrixDist)
invMatrixDist = 1/matrixDist
arrayDist = invMatrixDist.to_numpy()
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
        arrayPIndicesCopy = []
        arraySortIndices = []
        arraySelecao = []
        rowData = arrayData[row]
        arrayPMedia = means
        arrayPStds = stds
        arrayPCorr = arrayCorr[col]
        arrayPIndices = arrayIndices[col]
        arrayPDist = arrayDist[col]

        if np.isnan(arrayData[row,col]):
            precX = 0
            arrayPIndicesCopy = np.copy(arrayPIndices)
            arraySortIndices = np.sort(arrayPIndices)

            arraySelecao = arraySortIndices[-5:]

            arrayPos=[]
            for item in arraySelecao:
                position = np.where(arrayPIndicesCopy == item)[0][0]
                arrayPos.append(rowData[position])
                if method == "Mean":
                    precX = precX + ((1 / 5) * ((pMeans / arrayPMedia[position]) * rowData[position]))
                if method == "Correletion":
                    precX = precX + ((pStds / 5) * (((rowData[position] - arrayPMedia[position]) / arrayPStds[position]) * arrayPCorr[position]))
                if method == "InvDist":
                    somaDist=0
                    for item2 in arraySelecao:
                        position2 = np.where(arrayPIndicesCopy == item2)[0][0]
                        somaDist = somaDist + arrayPDist[position2]
                    precX = precX + ((arrayPDist[position]/somaDist) * rowData[position])
            if method == "Mean":
                arrayDataP[row,col] = precX
            if method == "Correletion":
                arrayDataP[row,col] = precX + pMedia
            if method == "InvDist":
                arrayDataP[row,col] = precX

df = pd.DataFrame(arrayDataP, columns = colNames, index = indexList)
df.to_csv('preenchida.csv')







