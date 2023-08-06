
from operator import index
import numpy as np
import pandas as pd
from copy import copy
import os

# Function to load data from a CSV file and return features (X), target (y), and header (column names)
def load_csv_data(input_csv_filename, mode="clf", verbose=False):
    """
    Loads data from a CSV file and returns X, y, and header.

    Args:
        input_csv_filename (str): The filename of the CSV file.
        mode (str, optional): The mode of the data. Can be "clf" (classification) or "regr" (regression).
                              Defaults to "clf".
        verbose (bool, optional): If True, prints information about the loaded data. Defaults to False.

    Returns:
        tuple: A tuple containing X (features), y (target), and header (column names).
    """
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(input_csv_filename)
    # Extract header (column names) from the DataFrame
    df_header = df.columns.values
    header = list(df_header)
    # Get the number of rows and columns in the DataFrame
    N, d = len(df), len(df_header) - 1
    # Extract features (X) by dropping the target column 'y'
    X = np.array(df.drop(['y'], axis=1))
    # Extract target (y) as a separate array
    y = np.array(df['y'])
    # Get the unique classes in the target (for classification mode)
    y_classes = list(set(y))
    # Check the shape of X and y to ensure consistency
    assert X.shape == (N, d)
    assert y.shape == (N,)
    # Check the data types of y based on the specified mode
    if mode == "clf":
        assert y.dtype in ['int64']  # check y are integers
    elif mode == "regr":
        assert y.dtype in ['int64', 'float64']  # check y are integers/floats
    else:
        exit("err: invalid mode given!")
    # Print information about the loaded data if verbose is True
    if verbose:
        print(" header={}\n X.shape={}\n y.shape={}\n len(y_classes)={}\n".format(header, X.shape, y.shape, len(y_classes)))
    return X, y, header

# Function for creating cross-validation folds and performing k-fold cross-validation
def cross_validate(model, X, y ,kfold=5, seed=1, model_name="non given"):
    """
    Performs k-fold cross-validation on a given model and data.

    Args:
        model (object): The machine learning model to be cross-validated.
        X (np.array): The features of the dataset.
        y (np.array): The target variable of the dataset.
        kfold (int, optional): The number of folds for cross-validation. Defaults to 5.
        seed (int, optional): The random seed for reproducibility. Defaults to 1.
        model_name (str, optional): The name of the model used for saving output files. Defaults to "non given".
    """
    # Function to create k cross-validation folds from the data indices
    def make_crossval_folds(N, kfold, seed=1):
        np.random.seed(seed)
        idx_all_permute = np.random.permutation(N)
        N_fold = int(N / kfold)
        idx_folds = []
        for i in range(kfold):
            start = i * N_fold
            end = min([(i + 1) * N_fold, N])
            idx_folds.append(idx_all_permute[start:end])
        return idx_folds

    N = len(X)
    idx_all = np.arange(0, N)
    # Create k cross-validation folds
    idx_folds = make_crossval_folds(N, kfold, seed=seed)
    assert len(idx_folds) == kfold

    print("\nCross-validating (kfold={}, seed={})...".format(kfold, seed))
    
    # Initialize variables to store average loss, R^2, and Pearson's correlation coefficients
    loss_train_avg, loss_val_avg = 0.0, 0.0
    r2_train_avg, r2_val_avg = 0.0, 0.0
    pearsonr_train_avg, pearsonr_val_avg = 0, 0
    cv_metrics_train = pd.DataFrame(columns=['loss_train', 'r2_train', 'pearsonr_train'])
    cv_metrics_val = pd.DataFrame(columns=['loss_val', 'r2_val', 'pearsonr_val'])
    cv_fold_preds = pd.DataFrame()
    cv_rest_preds = pd.DataFrame()

    # Loop through each fold and perform cross-validation
    for i in range(kfold):
        # Split data into training and validation sets
        idx_fold = idx_folds[i]
        idx_rest = np.delete(idx_all, idx_fold)
        X_rest, y_rest = X[idx_rest], y[idx_rest]
        X_fold, y_fold = X[idx_fold], y[idx_fold]

        # Create a copy of the model for training on the rest of the data
        model_rest = copy(model)
        model_rest.fit(X_rest, y_rest)

        # Make predictions on training and validation sets
        y_pred_rest = model_rest.predict(X_rest)
        y_pred_fold = model_rest.predict(X_fold)

        # Compute losses (e.g., mean squared error or cross-entropy)
        loss_train = model.loss(X_rest, y_rest)
        loss_val = model.loss(X_fold, y_fold)

        # Compute R^2 for training and validation sets
        r2_train = model.r2(X_rest, y_rest)
        r2_val = model.r2(X_fold, y_fold)

        # Compute Pearson's correlation coefficient for training and validation sets
        pearsonr_train = model.pearson(X_rest, y_rest)
        pearsonr_val = model.pearson(X_fold, y_fold)

        # Update average losses and metrics
        loss_train_avg += loss_train
        loss_val_avg += loss_val
        r2_train_avg += r2_train
        r2_val_avg += r2_val
        pearsonr_train_avg += pearsonr_train
        pearsonr_val_avg += pearsonr_val

        # Store metrics for each fold
        cv_metrics_train = cv_metrics_train.append([[loss_train, r2_train, pearsonr_train]], ignore_index=True)
        cv_metrics_val = cv_metrics_val.append([[loss_val, r2_val, pearsonr_val]], ignore_index=True)
        
        # Store predictions for each fold
        cv_fold_preds['fold' + str(i) + ' y'] = y_fold
        cv_fold_preds['fold' + str(i) + ' y_pred'] = y_pred_fold
        cv_rest_preds['fold' + str(i) + ' y'] = y_rest
        cv_rest_preds['fold' + str(i) + ' y_pred'] = y_pred_rest

        # Print results for each fold
        print(" [fold {}/{}] loss_train={:.6}, loss_validation={:.6}".format(i+1, kfold, loss_train, loss_val))

    # Calculate average metrics and losses over all folds
    loss_train_avg /= kfold
    loss_val_avg /= kfold
    r2_train_avg /= kfold
    r2_val_avg /= kfold
    pearsonr_train_avg /= kfold
    pearsonr_val_avg /= kfold

    # Store average metrics in a DataFrame
    cv_metrics_train.loc['Average'] = [loss_train_avg, r2_train_avg, pearsonr_train_avg]
    cv_metrics_val.loc['Average'] = [loss_val_avg, r2_val_avg, pearsonr_val_avg]
    
    # Save the metrics to CSV files
    cv_metrics_train.to_csv(model_name + "_metrics_train.csv")
    cv_metrics_val.to_csv(model_name + "_metrics_val.csv")
    cv_fold_preds.to_csv(model_name + "_fold_preds.csv")
    cv_rest_preds.to_csv(model_name + "_rest_preds.csv")

    # Print average results
    print("\nAverage metrics for {}-fold cross-validation (seed={}):".format(kfold, seed))
    print(" loss_train_avg={:.6}, loss_val_avg={:.6}, r2_train_avg={:.6}, r2_val_avg={:.6}, pearsonr_train_avg={:.6}, pearsonr_val_avg={:.6}".format(
        loss_train_avg, loss_val_avg, r2_train_avg, r2_val_avg, pearsonr_train_avg, pearsonr_val_avg))

# function for reading data from a excel file in a data folder
def read_data(file_name: str, format: str, index: bool =False) -> pd.DataFrame:
    """Reading csv or excel file from a data folder

    Args:
        file_name (str): name of the file
        format (str): format of the file. Can be csv or excel.
        index (bool): if the file has index

    Returns:
        pd.DataFrame: dataframe of the file
    """ 
    if format == 'csv': # read csv file
        return pd.read_csv(os.path.join('data',file_name + '.csv'), index_col = index)
    elif format == 'excel': # read excel file
        return pd.read_excel(os.path.join('data',file_name + '.xlsx'), index_col = index)
    else: # raise error if format is not csv or excel
        raise ValueError('Wrong format, please use csv or excel')
    
# function for saving data to a excel file or cvs in a data folder
def save_data(df: pd.DataFrame, file_name: str, format: str) -> None:
    """Save data to a excel file or cvs in a data folder

    Args:
        df (pd.DataFrame): dataframe to save
        file_name (str): name of the file
        format (str): format of the file. Can be csv or excel.
    """ 
    
    if format == 'csv': # save csv file
        df.to_csv(os.path.join('data',file_name + '.csv'), index = False)
    elif format == 'excel': # save excel file
        df.to_excel(os.path.join('data',file_name + '.xlsx'),index=False)
    else: # raise error if format is not csv or excel
        raise ValueError('Wrong format, please use csv or excel')
    return

def stat_describe (df: pd.DataFrame) -> pd.DataFrame:
    """This function describes the basic statistic charatristics of a dataframe.
    All features should be numbers.

    Args:
        df (pd.DataFrame): input pandas dataframe

    Returns:
        stat: pandas dataframe with basic statistic charatristics
    """    
    
    # import libraries
    import pandas as pd
    from scipy.stats import variation
    from scipy.stats import skew
    from scipy.stats import kurtosis

    # Getting count, mean, std, 25%, 50%, 75%, max, , skewness coef.
    stat = df.describe()

    # Get variation coef.
    temp = pd.DataFrame(variation(df.values)).T

    # Add variation coef. to stat
    temp.index = ['variation coef.']
    cols = list(stat.columns.values)
    temp.columns =cols
    stat = pd.concat([stat,temp])

    # Get skewness coef.
    temp = pd.DataFrame(skew(df.values)).T

    # Add skewness coef. to stat
    temp.index = ['skewness coef.']
    cols = list(stat.columns.values)
    temp.columns =cols
    stat = pd.concat([stat,temp])
    
    # get kurtosis coef.
    temp = pd.DataFrame(kurtosis(df.values)).T
    
    # Add kurtosis to stat
    temp.index = ['Kurtosis']
    cols = list(stat.columns.values)
    temp.columns =cols
    stat = pd.concat([stat,temp])

    stat = stat.round(2)
    

    return stat

def load_df_data(df) -> np.array:
    """functiom for converting df to array with last column as y and the rest as X and header

    Args:
        df (pd.DataFrame): input dataframe to be split
        

    Returns:
        X (np.array): all columns other than last
        y (np.array): last column
        header (np.array): name of the columns
    """    
    
    header = df.columns.values  # get the header
    X =df.values[:,0:df.shape[1]-1]  # except the last column
    y = df.values[:,df.shape[1]-1]  # only the last column
    return X, y, header

# function for scaling the data
def scale_data(X_train: np.array, X_test: np.array) -> np.array:
    """function for scaling the data

    Args:
        X_train (np.array): train data
        X_test (np.array): test data

    Returns:
        X_train_scaled (np.array): scaled train data
        X_test_scaled (np.array): scaled test data
    """    
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# function to check if a folder exist in a path. if not create it!
def check_folder_exist(path:str, folder_name:str):
    if not os.path.exists(os.path.join(path, folder_name)):
        os.mkdir(os.path.join(path, folder_name))
    return

def plot_corr_heatmap(df:pd.DataFrame, address:str, floating_point:str = '.1f') -> None:
    """This function draws and saves a correlation heatmap of a dataframe.
        inputes are dataframe and name of the saved file.

    Args:
        df (pd.DataFrame): input dataframe.
        address (str): where to save the file.
        floating_point (str, optional): number of points after floating point. Defaults to '.1f'.
    """    
    '''
    
    '''

    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")

    sns.set_style({'font.size':20,'font.family':'serif', 'font.serif':['Times New Roman'],'font.color':['black']})

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # set figure size
    f.set_size_inches(6.5, 6.5)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt=floating_point)
    plt.savefig(address, 
                dpi=300, bbox_inches='tight')
    return

def sigmoid(x):
    return 1/(1 + np.exp(-x))
# implement relevance function
# see paper: https://www.researchgate.net/publication/220699419_Utility-Based_Regression
def relevance(x):
    x = np.array(x)
    return sigmoid(x - 50)
# implement SMOTER
# see paper: https://core.ac.uk/download/pdf/29202178.pdf

def get_synth_cases(D, target, o=200, k=3, categorical_col = []):
    '''
    Function to generate the new cases.
    INPUT:
        D - pd.DataFrame with the initial data
        target - string name of the target column in the dataset
        o - oversampling rate
        k - number of nearest neighbors to use for the generation
        categorical_col - list of categorical column names
    OUTPUT:
        new_cases - pd.DataFrame containing new generated cases
    '''
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import KNeighborsRegressor
    new_cases = pd.DataFrame(columns = D.columns) # initialize the list of new cases 
    ng = o // 100 # the number of new cases to generate
    for index, case in D.iterrows():
        
        
        # find k nearest neighbors of the case
        knn = KNeighborsRegressor(n_neighbors = k+1) # k+1 because the case is the nearest neighbor to itself
        knn.fit(D.drop(columns = [target]).values, D[[target]])
        neighbors = knn.kneighbors(case.drop(labels = [target]).values.reshape(1, -1), return_distance=False).reshape(-1)
        neighbors = np.delete(neighbors, np.where(neighbors == index))
        for i in range(0, ng):
            # randomly choose one of the neighbors
            x = D.iloc[neighbors[np.random.randint(k)]]
            attr = {}          
            for a in D.columns:
                # skip target column
                if a == target:
                    continue;
                if a in categorical_col:
                    # if categorical then choose randomly one of values
                    if np.random.randint(2) == 0:
                        attr[a] = case[a]
                    else:
                        attr[a] = x[a]
                else:
                    # if continious column
                    diff = case[a] - x[a]
                    attr[a] = case[a] + np.random.randint(2) * diff
            # decide the target column
            new = np.array(list(attr.values()))
            d1 = cosine_similarity(new.reshape(1, -1), case.drop(labels = [target]).values.reshape(1, -1))[0][0]
            d2 = cosine_similarity(new.reshape(1, -1), x.drop(labels = [target]).values.reshape(1, -1))[0][0]
            attr[target] = (d2 * case[target] + d1 * x[target]) / (d1 + d2)
            
            # append the result
            new_cases = new_cases.append(attr,ignore_index = True)
                    
    return new_cases

def SmoteR(D, target, th = 0.999, o = 1000, u = 100, k = 3, categorical_col = []):
    '''
    The implementation of SmoteR algorithm:
    https://core.ac.uk/download/pdf/29202178.pdf
    INPUT:
        D - pd.DataFrame - the initial dataset
        target - the name of the target column in the dataset
        th - relevance threshold
        o - oversampling rate
        u - undersampling rate
        k - the number of nearest neighbors
    OUTPUT:
        new_D - the resulting new dataset
    '''
    # median of the target variable
    y_bar = D[target].median()
    
    # find rare cases where target less than median
    rareL = D[(relevance(D[target]) > th) & (D[target] > y_bar)]  
    # generate rare cases for rareL
    new_casesL = get_synth_cases(rareL, target, o, k , categorical_col)
    
    # find rare cases where target greater than median
    rareH = D[(relevance(D[target]) > th) & (D[target] < y_bar)]
    # generate rare cases for rareH
    new_casesH = get_synth_cases(rareH, target, o, k , categorical_col)
    
    new_cases = pd.concat([new_casesL, new_casesH], axis=0)
    
    # undersample norm cases
    norm_cases = D[relevance(D[target]) <= th]
    # get the number of norm cases
    nr_norm = int(len(norm_cases) * u / 100)
    
    norm_cases = norm_cases.sample(min(len(D[relevance(D[target]) <= th]), nr_norm))
    
    # get the resulting dataset
    new_D = pd.concat([new_cases, norm_cases], axis=0)
    
    return new_D

def sensitivity_analysis(model:object, X_train:np.array):
    # function to give a np.array of the sensitivity of each feature
    changes = list(range(-50, 51, 10))
    y = model.predict(X_train)
    results = []
    for i in range(X_train.shape[1]):
        results.append([])
        for j in changes:
            X_train_copy = X_train.copy()
            X_train_copy[:, i] = X_train_copy[:, i] * (1 + j/100)
            y_pred = model.predict(X_train_copy)
            # average of y_pred numpy array
            results[i].append(((y_pred - y).mean()) / y.mean() * 100)
    return results

def uncertainty_analysis (model:object, X:np.array, y:np.array) -> None:
    # function to give a np.array of the uncertainty of each feature
    y_pred = model.predict(X)

    err = y - y_pred
    err_mean = err.mean()
    err_std = err.std()
    print(f'mean error is {err_mean}')
    print(f'95% uncertainty bandwidth is {1.96 * err_std}')
    print(f'95% interval starts from {err_mean - 1.96 * err_std} and ends at {err_mean + 1.96 * err_std}')
    return 
