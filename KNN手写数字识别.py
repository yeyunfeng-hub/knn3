from os import listdir
from numpy import *
import numpy as np
import operator

def get_data(filename):
    data = []
    fr = open(filename)
    for i in range(32):
        line_Str = fr.readline()
        for j in range(32):
            data.append(int(line_Str[j]))
    return data

def get_laber(filename):
    label = int(filename.split('_')[0])
    return label

def KNN(timage,image,label,k):
    count = {}
    data_size = image.shape[0]
    distances = np.sqrt( np.sum(np.square(tile(timage,(data_size,1))-image),axis=1))
    sort_distances = distances.argsort()
    for i in range(k):
        min_laber = label[sort_distances[i]]
        count[min_laber] = count.get(min_laber,0)+1
    result = sorted(count.items(), key = operator.itemgetter(1), reverse = True)
    return result[0][0]


def trainingDataSet():
    label = []
    List = listdir('trainingDigits')
    m = len(List)
    image = zeros((m,1024))
    for i in range(m):
        str1 = List[i]
        label.append(get_laber(str1))
        image[i,:] = get_data('trainingDigits/%s' % str1)
    return label,image

#测试函数
def main():
    for k in [1,10,20,30,40,50,60,70,80,90,100]:
        Nearest_Neighbor_number = k
        label,image = trainingDataSet()
        testFileList = listdir('testDigits')
        error_num = 0
        test_number = len(testFileList)
        for i in range(test_number):
            filenameStr = testFileList[i]
            classNumStr = get_laber(filenameStr)
            timage = get_data('testDigits/%s' % filenameStr)
            classifierResult = KNN(timage, image, label, Nearest_Neighbor_number)
            if (classifierResult != classNumStr):
                error_num += 1.0
        print ("k=",k,":正确率:",100*(1-error_num/float(test_number)),'%')

if __name__ == "__main__":
    main()
