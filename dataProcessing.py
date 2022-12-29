import numpy as np
import cv2
from matplotlib import pyplot as plt


class dataProcessing :
    """Class that stores all the data and functions for reading
    and manipulation data

    Attributes:
    -----------------------------------------
    data:np.ndarray 
        data gained from reading the file
    
    rows:int 
        rows of the data
    
    cols:int
        cols of the data
    
    corrMat:np.ndarray
        Correlation Matrix of the data

    Methods:
    -----------------------------------------
    __loadData:
        read and load data into memory from specified file
    
    __getCorrMatrix:
        get correlation matrix of the data
    
    __getCovMatrix:
        get covariance matrix of the data

    __getBitmap:
        get bitmap of the matrix provided
    
    __displayImage_cv2:
        display an image using opencv-python
    
    __displayImage_plt:
        display an image using matplotlib.pyplot

    __getColorCoded:
        get the color coded image from a bitmap image
    
    __permuteMatrix:
        shuffle the given matrix row-wise
    
    __getSignatures:
        get the signatures of the given matrix of data

    __rearrange:
        sort the given data row-wise according to another set of data

    VisualizeData:
        visualize the correlation matrix in bitmap and colorcoded formats
        using matplotlib.pylot

    VisualizeSignatureTechnique:
        Visualize the permuted matrix, rearranged and colorcoded matrixes
        using matplotlib.pylot
    """



    def __init__(self, filename:str) -> None:
        """Constructor

        args:
            filename:str
            filename of the file to get data from

        returns:
            None
        """
        

        self.data = 0
        self.rows, self.cols = 0, 0
        self.__loadData(filename)
        self.corrMat = self.__getCorrMatrix(self.data)


    def __loadData(self, filename: str) -> None:
        """read and load data into memory from specified file

        args:
            filename:str
            name / path for the file to load data from

        returns:
            None 
        """


        with open(file=filename) as file: 
            self.rows = int(file.readline())
            self.cols = int(file.readline())
            file.readline()

            data = file.readlines()
            data = [x.split('\t') for x in data]
            self.data = np.array(data, dtype=np.float64)

        return data

    def __getCorrMatrix(self, Mat: np.ndarray) -> np.ndarray:
        """get correlation matrix of the data

            Correlation Formula:
                r[i,j] = C[i,j] / sqrt( C[i,i] * C[j,j] )
            where C is the covariance
    

            args:
                Mat:np.ndarray
                Matrix of data to create correlation matrix from

            returns:
                np.ndarray
                Correlation Matrix
        """


        covMat = self.__getCovMatrix(Mat)
        corrMat = np.zeros(covMat.shape, dtype=np.float64)

        for i in range(corrMat.shape[0]):
            for j in range(i+1):
                corrMat[i,j] = covMat[i,j] / np.sqrt(covMat[i,i] * covMat[j,j])
                corrMat[j,i] = covMat[i,j] / np.sqrt(covMat[i,i] * covMat[j,j])
        
        return corrMat


    def __getCovMatrix(self, Mat: np.ndarray) -> np.ndarray:
        """get covariance matrix of the data

            Covariance Formula:
                C[x,y] = ( stddev(x) * stddev(y) ) / N

            args:
                Mat:np.ndarray
                Matrix of the data to create covariance matrix

            returns:
                np.ndarray
                Covariance matrix
        """
        

        x_means = np.mean(Mat, axis=1)
        xc = np.ndarray(Mat.shape, dtype=np.float64)

        for i in range(Mat.shape[0]):
            xc[i] = Mat[i] - x_means[i]

        covMat = (xc @ xc.T) / self.rows


        return covMat

    def __getBitmap(self, Mat: np.ndarray, mm: bool = 1) -> np.ndarray:
        """get bitmap of the matrix provided

            args:
                Mat:np.ndarray
                Matrix of values of any size

                mm:bool = 1 (default)
                threshhold is mean of Mat if mm = True
                else threshhold is the median of Mat
                set to True by default
            
            returns:
                np.ndarray
                bitmap of the Matrix provided
        """
        

        threshold = 0
        if mm == 1:
            threshold = np.mean(Mat)
        else:
            threshold = np.median(Mat)

        # threshold -= 0.05

        m = np.zeros(Mat.shape, dtype=np.float64)
        m[Mat < threshold] = 1

        return m

    def __displayImage_cv2(self, name: str,  Mat: np.ndarray) -> None:
        """display an image using opencv-python

            args:
                name:str
                title / name of the image

                Mat:np.ndarray
                Matrix to display
                can be of shape (m,n) for bitmap images
                can be of shape (m,n,bgr) for colored images 

            returns:
                None
        """


        cv2.imshow(name, Mat)
        cv2.waitKey(0)


    def displayImage_plt(self, name:str, Mat: np.ndarray) -> None:
        """display an image using matplotlib.pyplot

        
            args:
                name:str
                title / name of the image

                Mat:np.ndarray
                Matrix to display
                can be of shape (m,n) for bitmap images
                can be of shape (m,n,bgr) for colored images 

            returns:
                None
        """


        plt.title(name)
        plt.imshow(Mat)
        plt.show()
        plt.clf()


    def __getColorCoded(self, Mat: np.ndarray, color: tuple = (0,1,0)) -> np.ndarray:
        """get the color coded image from a bitmap image

            args:
                Mat:np.ndarray
                Matrix of image in bitmap

                color:tuple = (0,1,0) (default)
                color in bgr
        """


        colored_m = np.zeros((Mat.shape[0], Mat.shape[1], 3))

        for i in range(Mat.shape[0]):
            for j in range(i+1):
                colored_m[i,j] = Mat[i,j] * np.asarray(color) / np.max(Mat.T[j])
                colored_m[j,i] = Mat[i,j] * np.asarray(color) / np.max(Mat.T[i])

        return colored_m


    def __permuteMatrix(self, Mat: np.ndarray) -> np.ndarray:
        """shuffle the given matrix row-wise
            use of np.random.shuffle to shuffle the matrix

            args:
                Mat:np.ndarray
                Matrix to permute

            returns:
                np.ndarray
                permutated matrix
        """
        

        m = Mat.copy()
        np.random.shuffle(m)
        return m


    def __getSignatures(self, Mat: np.ndarray) -> np.ndarray:
        """get the signatures of the given matrix of data

            steps to get signatures:
                o Sum all the values in a row
                o Calculate mean of the row
                o Multiply the Sum of the row with its Mean

            args:
            `   Mat:np.ndarray
                Matrix of data to produce signatures from
            
            returns:
                np.ndarray:
                    array containing the signatures of data
        """


        row_sum = np.sum(Mat, axis=1)
        row_means = np.mean(Mat, axis=1)

        signature = row_sum * row_means
        return signature

    
    def __rearrange(self, permMat: np.ndarray, signatures: np.ndarray) -> np.ndarray:
        """sort the given data row-wise according to another set of data

            uses np.argsort to get the ranks of sigantures wrt other values of the matrix
            then rearrange the permutated matrix according to the rank using the builtin method of python

            args:
                permMat:np.ndarray
                permutated matrix to rearrange

                signatures: np.ndarray
                signatures of data
            
            returns:
                np.ndarray
                re-arranged matrix of the permutated data
        """


        unshuffle = permMat[np.argsort(signatures)]

        return unshuffle

    
    def VisualizeData(self):
        """visualize the correlation matrix in bitmap and colorcoded formats
        using matplotlib.pylot

            args:
                None
            
            returns:
                None
        """


        fig = plt.figure()
        fig.suptitle("Visualization of Data, Task # 1")
        
        ax = fig.add_subplot(1,2,1)
        plt.imshow(self.__getBitmap(self.corrMat), cmap= 'gray')
        ax.set_title("Bitmap of Correlation Matrix")

        ax = fig.add_subplot(1,2,2)
        plt.imshow(self.__getColorCoded(self.corrMat))
        ax.set_title("Color-coded Correlation Matrix")

        plt.show()
        plt.clf()

    def VisualizeSignatureTechnique(self):
        """Visualize the permuted matrix, rearranged and colorcoded matrixes
        using matplotlib.pylot

            args:
                None

            returns:
                None
        """


        fig = plt.figure(figsize=(12,5))
        plt.suptitle("Visualization of Signature Techique to unshuffle data, Task # 2")

        perm_data = self.__permuteMatrix(self.data)
        perm_corr = self.__getCorrMatrix(perm_data)
        perm_bmp = self.__getBitmap(perm_corr)

        data_signatures = self.__getSignatures(perm_data)
        unshuffled_data = self.__rearrange(perm_data, data_signatures)
        unshuffled_corr = self.__getCorrMatrix(unshuffled_data)
        unshuffled_colored = self.__getColorCoded(unshuffled_corr)

        ax = fig.add_subplot(1,3,1)
        plt.imshow(self.__getBitmap(self.corrMat), cmap= 'gray')
        ax.set_title("Bitmap of original data")

        ax = fig.add_subplot(1,3,2)
        plt.imshow(perm_bmp, cmap= 'gray')
        ax.set_title("Bitmap of data after permutation")

        ax = fig.add_subplot(1,3,3)
        plt.imshow(unshuffled_colored, cmap= 'gray')
        ax.set_title("ColorCoded Image of data after \nsignature technique and arrangement")

        plt.show()
        plt.clf()