# Import necessary libraries
import numpy as np
from copy import deepcopy
from graphviz import Digraph

# Define the ModelTree class
class ModelTree(object):

    # Initialize the ModelTree object with specified parameters
    def __init__(self, model, max_depth=5, min_samples_leaf=10,
                 search_type="greedy", n_search_grid=100):

        # Initialize model and hyperparameters
        self.model = model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.search_type = search_type
        self.n_search_grid = n_search_grid
        self.tree = None

    # Get the parameters of the ModelTree
    def get_params(self, deep=True) -> dict:
        """
        Returns the parameters of the ModelTree.

        Args:
            deep (bool, optional): Whether to return the parameters of the underlying model (deep=True)
                                   or the model itself (deep=False). Defaults to True.

        Returns:
            dict: ModelTree parameters.
        """        
        return {
            "model": self.model.get_params() if deep else self.model,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "search_type": self.search_type,
            "n_search_grid": self.n_search_grid,
        }

    # Set the parameters of the ModelTree
    def set_params(self, **params: dict):
        """
        Update the parameters of the ModelTree.

        Returns:
            object: ModelTree object with updated parameters.
        """        
        for param, value in params.items():
            setattr(self, param, value)
        return self

    # Represent the ModelTree object as a string
    def __repr__(self):
        class_name = self.__class__.__name__
        return "{}({})".format(class_name, ', '.join(["{}={}".format(k, v) for k, v in self.get_params(deep=False).items()]))

    # Fit the ModelTree to the training data
    def fit(self, X, y, verbose=False):

        # Get hyperparameters
        model = self.model
        min_samples_leaf = self.min_samples_leaf
        max_depth = self.max_depth
        search_type = self.search_type
        n_search_grid = self.n_search_grid

        if verbose:
            print(" max_depth={}, min_samples_leaf={}, search_type={}...".format(max_depth, min_samples_leaf, search_type))

        # Helper function to build the tree recursively
        def _build_tree(X, y):

            global index_node_global

            # Create a node for the tree
            def _create_node(X, y, depth, container):

                # Fit the model to the node data and calculate loss
                loss_node, model_node = _fit_model(X, y, model)
                node = {"name": "node",
                        "index": container["index_node_global"],
                        "loss": loss_node,
                        "model": model_node,
                        "data": (X, y),
                        "n_samples": len(X),
                        "j_feature": None,
                        "threshold": None,
                        "children": {"left": None, "right": None},
                        "depth": depth}
                container["index_node_global"] += 1

                return node

            # Recursively split node and traverse node until a terminal node is reached
            def _split_traverse_node(node, container):

                # Perform split and collect result
                result = _splitter(node, model,
                                   max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   search_type=search_type,
                                   n_search_grid=n_search_grid)

                # Return terminal node if split is not advised
                if not result["did_split"]:
                    if verbose:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(" {}*leaf {} @ depth {}: loss={:.6f}, N={}".format(depth_spacing_str, node["index"], node["depth"], node["loss"], result["N"]))
                    return

                # Update node information based on splitting result
                node["j_feature"] = result["j_feature"]
                node["threshold"] = result["threshold"]
                del node["data"]  # delete node stored data

                # Extract splitting results
                (X_left, y_left), (X_right, y_right) = result["data"]
                model_left, model_right = result["models"]

                # Report created node to the user
                if verbose:
                    depth_spacing_str = " ".join([" "] * node["depth"])
                    print(" {}node {} @ depth {}: loss={:.6f}, j_feature={}, threshold={:.6f}, N=({},{})".format(depth_spacing_str, node["index"], node["depth"], node["loss"], node["j_feature"], node["threshold"], len(X_left), len(X_right)))

                # Create children nodes
                node["children"]["left"] = _create_node(X_left, y_left, node["depth"]+1, container)
                node["children"]["right"] = _create_node(X_right, y_right, node["depth"]+1, container)
                node["children"]["left"]["model"] = model_left
                node["children"]["right"]["model"] = model_right

                # Split nodes
                _split_traverse_node(node["children"]["left"], container)
                _split_traverse_node(node["children"]["right"], container)

            container = {"index_node_global": 0}  # mutable container
            root = _create_node(X, y, 0, container)  # depth 0 root node
            _split_traverse_node(root, container)  # split and traverse root node

            return root

        # Construct the tree
        self.tree = _build_tree(X, y)

    # Predict using the trained ModelTree
    def predict(self, X):

        assert self.tree is not None
        def _predict(node, x):
            # Check if node is a leaf (no children)
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:
                y_pred_x = node["model"].predict([x])[0]  # Predict using the model at the leaf node
                return y_pred_x
            else:
                if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                    return _predict(node["children"]["left"], x)  # Recurse left
                else:  # x[j] > threshold
                    return _predict(node["children"]["right"], x)  # Recurse right

        y_pred = np.array([_predict(self.tree, x) for x in X])
        return y_pred

    # Explain the prediction of the ModelTree
    def explain(self, X, header):

        assert self.tree is not None
        def _explain(node, x, explanation):
            # Check if node is a leaf (no children)
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:
                return explanation
            else:
                if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                    explanation.append("{} = {:.6f} <= {:.6f}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                    return _explain(node["children"]["left"], x, explanation)  # Recurse left
                else:  # x[j] > threshold
                    explanation.append("{} = {:.6f} > {:.6f}".format(header[node["j_feature"]], x[node["j_feature"]], node["threshold"]))
                    return _explain(node["children"]["right"], x, explanation)  # Recurse right

        explanations = [_explain(self.tree, x, []) for x in X]
        return explanations

    # Calculate mean squared error loss for the ModelTree
    def loss(self, X, y):
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y, self.predict(X))

    # Calculate R-squared for the ModelTree
    def r2(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

    # Calculate Pearson correlation coefficient for the ModelTree
    def pearson(self, X, y):
        from scipy.stats import pearsonr
        return pearsonr(y, self.predict(X))[0]

    # Export the ModelTree as a Graphviz diagram
    def export_graphviz(self, output_filename, feature_names,
                        export_png=True, export_pdf=True):
        """
        Assumes node structure of:
           node["name"]
           node["n_samples"]
           node["children"]["left"]
           node["children"]["right"]
           node["j_feature"]
           node["threshold"]
           node["loss"]
        """
        g = Digraph('g', node_attr={'shape': 'record', 'height': '.1'})

        def build_graphviz_recurse(node, parent_node_index=0, parent_depth=0, edge_label=""):

            # Empty node
            if node is None:
                return

            # Create node
            node_index = node["index"]
            if node["children"]["left"] is None and node["children"]["right"] is None:
                threshold_str = ""
            else:
                threshold_str = "{} <= {:.1f}\\n".format(feature_names[node['j_feature']], node["threshold"])

            label_str = "{} n_samples = {}\\n loss = {:.6f}".format(threshold_str, node["n_samples"], node["loss"])

            # Create node
            nodeshape = "rectangle"
            bordercolor = "black"
            fillcolor = "white"
            fontcolor = "black"
            g.attr('node', label=label_str, shape=nodeshape)
            g.node('node{}'.format(node_index),
                   color=bordercolor, style="filled",
                   fillcolor=fillcolor, fontcolor=fontcolor)

            # Create edge
            if parent_depth > 0:
                g.edge('node{}'.format(parent_node_index),
                       'node{}'.format(node_index), label=edge_label)

            # Traverse child or append leaf value
            build_graphviz_recurse(node["children"]["left"],
                                   parent_node_index=node_index,
                                   parent_depth=parent_depth + 1,
                                   edge_label="")
            build_graphviz_recurse(node["children"]["right"],
                                   parent_node_index=node_index,
                                   parent_depth=parent_depth + 1,
                                   edge_label="")

        # Build graph
        build_graphviz_recurse(self.tree,
                               parent_node_index=0,
                               parent_depth=0,
                               edge_label="")

        # Export pdf
        if export_pdf:
            print("Saving model tree diagram to '{}.pdf'...".format(output_filename))
            g.format = "pdf"
            g.render(filename=output_filename, view=False, cleanup=True)

        # Export png
        if export_png:
            print("Saving model tree diagram to '{}.png'...".format(output_filename))
            g.format = "png"
            g.render(filename=output_filename, view=False, cleanup=True)


# ***********************************
#
# Side functions
#
# ***********************************


def _splitter(node, model,
              max_depth=5, min_samples_leaf=10,
              search_type="greedy", n_search_grid=100):

    # Extract data
    X, y = node["data"]
    depth = node["depth"]
    N, d = X.shape

    # Find feature splits that might improve loss
    did_split = False
    loss_best = node["loss"]
    data_best = None
    models_best = None
    j_feature_best = None
    threshold_best = None

    # Perform threshold split search only if node has not hit max depth
    if (depth >= 0) and (depth < max_depth):

        for j_feature in range(d):

            # If using adaptive search type, decide on one to use
            search_type_use = search_type
            if search_type == "adaptive":
                if N > n_search_grid:
                    search_type_use = "grid"
                else:
                    search_type_use = "greedy"

            # Use decided search type and generate threshold search list (j_feature)
            threshold_search = []
            if search_type_use == "greedy":
                for i in range(N):
                    threshold_search.append(X[i, j_feature])
            elif search_type_use == "grid":
                x_min, x_max = np.min(X[:, j_feature]), np.max(X[:, j_feature])
                dx = (x_max - x_min) / n_search_grid
                for i in range(n_search_grid+1):
                    threshold_search.append(x_min + i*dx)
            else:
                exit("err: invalid search_type = {} given!".format(search_type))

            # Perform threshold split search on j_feature
            for threshold in threshold_search:

                # Split data based on threshold
                (X_left, y_left), (X_right, y_right) = _split_data(j_feature, threshold, X, y)
                N_left, N_right = len(X_left), len(X_right)

                # Splitting conditions
                split_conditions = [N_left >= min_samples_leaf,
                                    N_right >= min_samples_leaf]

                # Do not attempt to split if split conditions not satisfied
                if not all(split_conditions):
                    continue

                # Compute weight loss function
                loss_left, model_left = _fit_model(X_left, y_left, model)
                loss_right, model_right = _fit_model(X_right, y_right, model)
                loss_split = (N_left*loss_left + N_right*loss_right) / N

                # Update best parameters if loss is lower
                if loss_split < loss_best:
                    did_split = True
                    loss_best = loss_split
                    models_best = [model_left, model_right]
                    data_best = [(X_left, y_left), (X_right, y_right)]
                    j_feature_best = j_feature
                    threshold_best = threshold

    # Return the best result
    result = {"did_split": did_split,
              "loss": loss_best,
              "models": models_best,
              "data": data_best,
              "j_feature": j_feature_best,
              "threshold": threshold_best,
              "N": N}

    return result

def _fit_model(X, y, model):

    model_copy = deepcopy(model)  # must deepcopy the model!
    model_copy.fit(X, y)
    y_pred = model_copy.predict(X)
    loss = model_copy.loss(X, y)
    assert loss >= 0.0

    return loss, model_copy

def _split_data(j_feature, threshold, X, y):
    
    idx_left = np.where(X[:, j_feature] <= threshold)[0]
    idx_right = np.delete(np.arange(0, len(X)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X)
    return (X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])


# Import necessary modules
from src.ModelTree import ModelTree
import pandas as pd
import os

# Define the class DevelopModel for developing regression models
class DevelopModel:
    
    def __init__(self, leaf_model_name: str = 'linear', model_tree: bool = False, max_depth: int = 2,
                 min_samples_leaf: int = 2, search_type: str = 'grid', n_search_grid: int = 10):
        """
        Initialize the DevelopModel class.

        Parameters:
            leaf_model_name (str): The name of the leaf model to use (e.g., 'linear', 'lasso', etc.).
            model_tree (bool): True to use ModelTree (tree-based ensemble), False for a standalone regression model.
            max_depth (int): Maximum depth of the model tree (used only if model_tree=True).
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node (used only if model_tree=True).
            search_type (str): The type of search used for finding the best feature split (used only if model_tree=True).
            n_search_grid (int): Number of grid points for feature split search (used only if model_tree=True and search_type='grid').
        """
        self.leaf_model_name = leaf_model_name  # Leaf name if it is a model tree; otherwise, it is the model name.
        self.max_depth = max_depth  # Maximum model tree depth.
        self.min_samples_leaf = min_samples_leaf  # Minimum model tree leaf size.
        self.search_type = search_type  # Model tree search type.
        self.n_search_grid = n_search_grid  # Number of grid points for model tree search (used only if search_type='grid').
        self.leaf_model = self.leaf_model_fetcher(leaf_model_name)  # Leaf model if it is a model tree; otherwise, it is the model.
        self.model_tree = model_tree  # Boolean flag indicating whether it's a model tree or not.

        # Initialize the appropriate model based on the model_tree flag
        if self.model_tree:
            self.model_name = 'm5p_' + leaf_model_name + '_tree'
            self.model = ModelTree(self.leaf_model, self.max_depth, self.min_samples_leaf,
                                   self.search_type, self.n_search_grid)
        else:
            self.model_name = self.leaf_model_name
            self.model = self.leaf_model

    # Function to fetch the appropriate leaf model based on the model_name
    def leaf_model_fetcher(self, model_name: str) -> object:
        """
        Fetches the appropriate leaf model based on the model_name.

        Parameters:
            model_name (str): Name of the leaf model to fetch.

        Returns:
            object: The instantiated leaf model object.
        """
        # List of existing model names
        existing_models_name = ['linear', 'lasso', 'ridge', 'DecisionTree', 'logistic', 'randomforest', 'NN', 'svm', 'gaussianprocess']

        # Check if the given model_name is valid
        if model_name not in existing_models_name:
            raise ValueError('Wrong model name, please use one of the following: ' + str(existing_models_name))

        # Instantiate the appropriate leaf model based on the model_name
        if model_name == 'linear':
            from models.linear_regr import linear_regr
            model = linear_regr()
        elif model_name == 'lasso':
            from models.lasso_regr import lasso_regr
            model = lasso_regr()
        elif model_name == 'ridge':
            from models.ridge_regr import ridge_regr
            model = ridge_regr()
        elif model_name == 'DecisionTree':
            from models.rt_regr import rt_regr
            model = rt_regr()
        elif model_name == 'logistic':
            from models.logistic_regr import logistic_regr
            model = logistic_regr()
        elif model_name == 'randomforest':
            print('randomforest is not implemented yet')
            # Note: The randomforest_regr model is commented out, indicating that it's not yet implemented.
            # You can uncomment the relevant lines and implement the randomforest_regr model if needed.
        elif model_name == 'NN':
            from models.NN_regr import NN_regressor
            model = NN_regressor()
        elif model_name == 'svm':
            from models.svm_regr import svm_regr
            model = svm_regr()
        elif model_name == 'gaussianprocess':
            from models.gp_regr import gp_regr
            model = gp_regr()

        return model

    # Function to check if a folder exists in a path. If not, create it.
    def check_folder_exist(self, path: str, folder_name: str):
        """
        Checks if a folder exists in the specified path. If not, creates the folder.

        Parameters:
            path (str): The base path where the folder should be checked or created.
            folder_name (str): The name of the folder to check or create.
        """
        if not os.path.exists(os.path.join(path, folder_name)):
            os.mkdir(os.path.join(path, folder_name))
        return

    # Fit the model to the given training data (X, y).
    def fit(self, X: np.array, y: np.array, verbose: bool = False) -> None:
        """
        Fits the model to the given training data (X, y).

        Parameters:
            X (np.array): The feature matrix of shape (n_samples, n_features).
            y (np.array): The target vector of shape (n_samples,).
            verbose (bool): If True, print verbose information during the fitting process.
        """
        if self.model_tree:
            self.model.fit(X, y, verbose=verbose)
        else:
            self.model.fit(X, y)
        return

    # Predict the target values for the given input features (X).
    def predict(self, X: np.array) -> None:
        """
        Predicts the target values for the given input features (X).

        Parameters:
            X (np.array): The feature matrix of shape (n_samples, n_features).

        Returns:
            np.array: The predicted target values of shape (n_samples,).
        """
        return self.model.predict(X)

    # Export the model tree as a Graphviz diagram and save it to a file.
    def export_graph(self, header: list) -> None:
        """
        Exports the model tree as a Graphviz diagram and saves it to a file.

        Parameters:
            header (list): List of feature names to use for node explanations in the graph.
        """
        self.check_folder_exist("output", self.model_name)
        self.model.export_graphviz(os.path.join("output", self.model_name, "model_tree_" + self.model_name), header,
                                   export_png=True, export_pdf=False)
        return

    # Print evaluation metrics for the model using test data.
    def evaluate(self, X_test: np.array, y_test: np.array) -> None:
        """
        Evaluates the performance of the model on the given test data (X_test, y_test).

        Parameters:
            X_test (np.array): The feature matrix of shape (n_samples, n_features) for testing.
            y_test (np.array): The true target values of shape (n_samples,) for testing.
        """
        print(f'\n{self.model_name} loss for test data is {self.model.loss(X_test, y_test)}')
        print(f'{self.model_name} r2 for test data is {self.model.r2(X_test, y_test)}')
        print(f'{self.model_name} pearsonr for test data is {self.model.pearson(X_test, y_test)}')
        return

    # Save model outputs (predicted values) for both training and testing data to output files.
    def save_output(self, X_train, y_train, X_test, y_test) -> None:
        """
        Saves the model outputs (predicted values) for both training and testing data to output files.

        Parameters:
            X_train (np.array): The feature matrix of shape (n_samples, n_features) for training.
            y_train (np.array): The true target values of shape (n_samples,) for training.
            X_test (np.array): The feature matrix of shape (n_samples, n_features) for testing.
            y_test (np.array): The true target values of shape (n_samples,) for testing.
        """
        self.check_folder_exist("output", self.model_name)

        # Create a dataframe to save the train output
        result = pd.DataFrame()
        result['y_train'] = y_train
        result['y_train_pred'] = self.model.predict(X_train)
        # Save the train output
        result.to_excel(os.path.join('output', self.model_name, self.model_name + '_train.xlsx'))

        # Create a dataframe to save the test output
        result = pd.DataFrame()
        result['y_test'] = y_test
        result['y_test_pred'] = self.model.predict(X_test)
        # Save the test output
        result.to_excel(os.path.join('output', self.model_name, self.model_name + '_test.xlsx'))

    # Save the model as a pickle file in the output folder.
    def save_model(self) -> None:
        """
        Saves the model as a pickle file in the output folder.
        """
        self.check_folder_exist("output", self.model_name)
        import pickle
        # Save the model
        with open(os.path.join('output', self.model_name, self.model_name + '.pkl'), 'wb') as outp:
            pickle.dump(self.model, outp, pickle.HIGHEST_PROTOCOL)
        return

    # Load the model from a previously saved pickle file.
    def load_model(self) -> None:
        """
        Loads the model from a previously saved pickle file.
        """
        self.check_folder_exist("output", self.model_name)
        import pickle
        # Load the model
        with open(os.path.join('output', self.model_name, self.model_name + '.pkl'), 'rb') as inp:
            self.model = pickle.load(inp)
        return

    # Function to perform cross-validation.
    def cross_validate(self, X: np.array, y: np.array, kfold: int = 5, seed: int = 13) -> None:
        """
        Function to perform cross-validation.

        Parameters:
            X (np.array): The feature matrix of shape (n_samples, n_features).
            y (np.array): The target vector of shape (n_samples,).
            kfold (int): Number of cross-validation folds.
            seed (int): Random seed for reproducibility.
        """
        from src.utils import cross_validate
        cross_validate(self.model, X, y, kfold=5, seed=13, model_name=self.model_name)
        return
