import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class simpleLinearRegression:
    """
    Class for Simple Linear Regression. This solves the equation:
        y=b1x+b0
    Where b1 is:
        b1 = SUM((Xi - Xbar) * (Yi - Ybar))/SUM((Xi - Xbar)^2)

    ...

    Attributes
    ----------
    b0 : float
        Represents the Y intercept
    b1 : float
        Represents the Slope

    Methods
    -------
    slope(X, Y)
        Calculates the slope of two vectors
    yIntercept(slope, xBar, yBar)
        Calculates the Y intercept
    train(X,Y)
        Calculates the slope and y Intercept
    printEquation()
        Prints the equation of the line
    predict(X)
        Generates a set of predictions for a given vector
    mse(yPred, yActual)
        Calculate the Mean Squared Error
    plot(yPred)
        Plots data points
    """

    def __init__(self):
        self.b0 = None # Y intercept
        self.b1 = None # Slope

    def slope(self, X, Y):
        """
        Given a vector of X and Y, calculate the slope

        Parameters
        ----------
        X : Numpy Array
            Array of inputs
        Y : Numpy Array
            Array of Outputs
        """
        numerator = sum((X-X.mean()) * (Y-Y.mean()))
        denominator = sum((X-X.mean())**2)
        return numerator/denominator

    def yIntercept(self, slope, xBar, yBar):
        """
        Given a slope, X and Y, calculate the Y-Intercept

        yIntercept = yBar - slope*xBar

        Parameters
        ----------
        slope : float
            Slope of line
        X : float
            Mean of all X values
        Y : float
            Mean of all Y values
        """
        return yBar - slope*xBar

    def train(self, X, Y):
        """
        Get the coeffiecents for a simple Linear Regression Problem

        Parameters
        ----------
        X : numpy Array
            X inputs
        Y : numpy array
            Y inputs
        """
        self.b1 = self.slope(X, Y)
        self.b0 = self.yIntercept(self.b1, X.mean(), Y.mean())

    def printEquation(self):
        """Print out the equation if it exists"""
        if self.b1 is not None and self.b0 is not None:
            print(f"Eqaution:\n\tY = {self.b1} * x + {self.b0}")
        else:
            print("[ERROR] - No equation found.")

    def predict(self, X):
        """
        Use the equation to make a new prediction

        Parameters
        ----------
        X : numpy Array
            Array of variables to predict for.
        """
        if self.b1 is not None and self.b0 is not None:
            return self.b1 * X + self.b0
        else:
            print("[ERROR] - No equation found.")

    def mse(self, yPred, yActual):
        """
        Calculate the Mean Squared Error

        SUM((yPred - yActual)^2) / N

        Parameters
        ----------
        yPred : numpy Array
            Predicted outputs
        yActual : numpy array
            Actual Outputs
        """
        return sum((yPred-yActual)**2) / yPred.shape[0]

    def plot(self, X, Y):
        """
        Plots data points

        Parameters
        ----------
        X : numpy Array
            Array of X values
        Y : numpy array
            Array of Y values
        """
        # Plot data points
        plt.scatter(X, Y)

        # Plot line
        x = np.linspace(X.min(), X.max(), X.shape[0])
        plt.plot(x, self.b1*x+self.b0, linestyle='solid', color='red')

        # Add labels and show
        plt.title('Weight vs Height')
        plt.xlabel('Height')
        plt.ylabel('Weight')
        plt.show()

if __name__ == '__main__':
    """Run Simple Linear Regression to predict the weight of Male based on their Height"""

    df = pd.read_csv("weight-height.csv")
    # Grab all males for to ensure we have a better correlation
    df = df[df['Gender'] == 'Male']
    X = df['Height']
    Y = df['Weight']

    # Split into train and testing
    # 2/3 training... 1/3 testing
    Xtrain = X[:X.shape[0]*2//3]
    Xtest = X[X.shape[0]*2//3:]
    Ytrain = Y[:Y.shape[0]*2//3]
    Ytest = Y[Y.shape[0]*2//3:]

    # Build Model
    linearReg = simpleLinearRegression()
    linearReg.train(Xtrain,Ytrain)

    # Print out equation
    linearReg.printEquation()

    # Make predictions
    yPred = linearReg.predict(Xtest)

    # Evaluate predictions
    error = linearReg.mse(yPred, Ytest)

    print("Mean Squared Error: ", error)
    print("Root Mean Squared Error: ", error**0.5)

    linearReg.plot(Xtest, Ytest)
