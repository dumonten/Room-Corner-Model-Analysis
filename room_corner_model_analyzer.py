import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import stats
import os
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score 


class RoomCornerModelAnalyzer: 
    @staticmethod
    def draw_plots(json_source: str, dir_to_save:str="./plots") -> list[str]:
        """
        Draw various plots based on the data provided in a JSON file.

        Parameters
        ----------
        json_source : str
            Path to the JSON file containing the data.
        dir_to_save : str, optional
            Directory path where the plots will be saved. Default is "./plots".

        Returns
        -------
        list[str]
            Returns paths to all saved plots images. 
        """
            
        data = pd.read_json(json_source)
         
        if not os.path.exists(dir_to_save): 
            os.mkdir(dir_to_save)
        elif not os.path.isdir(dir_to_save): 
            raise ValueError("Error: You passed an existing path not to a directory.")            
        
        plots_paths = []
        

        # Confusion Matrix to compare the number of corners predicted by the model 
        # with the ground true number of corners
        path = __class__.confusion_matrix(data, dir_to_save)
        plots_paths.append(path)


        accuracy = pd.DataFrame(columns=['MAE', 'MSE', 'R2'])
        # Calculate the Mean Absolute Error (MAE) between ground truth corners and predicted corners
        mae = mean_absolute_error(data["gt_corners"], data["rb_corners"])
        # Calculate the Mean Squared Error (MSE) between ground truth corners and predicted corners
        mse = mean_squared_error(data["gt_corners"], data["rb_corners"])        
        # Calculate the R-squared score (coefficient of determination) between ground truth corners and predicted corners
        r2 = r2_score(data["gt_corners"], data["rb_corners"])
        accuracy.loc[0] = [mae, mse, r2]
        
        path = dir_to_save + "/accuracy_table.html"
        accuracy.to_html(path)
        plots_paths.append(path)
        

        data_means = data[['mean', 'floor_mean', 'ceiling_mean']]
        # Generate a box plot for the mean values and store the path to the plot
        path = __class__.box_plot(data_means, "means", "Box Plots for average mean, floor mean and ceiling mean", dir_to_save)
        plots_paths.append(path)

        data_maxs = data[['max', 'floor_max', 'ceiling_max']]
        # Generate a box plot for the max values and store the path to the plot
        path = __class__.box_plot(data_maxs, "maxs", "Box Plots for average max, floor max and ceiling max", dir_to_save)
        plots_paths.append(path)

        data_mins = data[['min', 'floor_min', 'ceiling_min']]
        # Generate a box plot for the min values and store the path to the plot
        path = __class__.box_plot(data_mins, "mins", "Box Plots for average min, floor min and ceiling min", dir_to_save)
        plots_paths.append(path)

        data_average = data[['mean', 'max', 'min']]
        # Generate a box plot for the average deviation values and store the path to the plot
        path = __class__.box_plot(data_average, "average_deviation_type", "Box Plots for deviation values of average", dir_to_save)
        # Append the path of the generated plot to the list of plot paths
        plots_paths.append(path)

        data_floor = data[['floor_mean', 'floor_max', 'floor_min']]
        # Generate a box plot for the floor deviation values and store the path to the plot
        path = __class__.box_plot(data_floor, "floor_deviation_type", "Box Plots for deviation values of floor", dir_to_save)
        plots_paths.append(path)

        data_ceiling = data[['ceiling_mean', 'ceiling_max', 'ceiling_min']]
        # Generate a box plot for the ceiling deviation values and store the path to the plot
        path = __class__.box_plot(data_ceiling, "ceiling_deviation_type", "Box Plots for deviation values of ceiling", dir_to_save)
        plots_paths.append(path)

        
        metrics = pd.DataFrame(columns=['MAD', 'MAPD', 'RMSD', 'Emax'])

        # Calculate metrics for the average mean values and store them in the DataFrame
        average_mean = __class__.get_metrics(data['mean'])
        metrics.loc['average_mean'] = average_mean

        # Calculate metrics for the floor mean values and store them in the DataFrame
        floor_mean = __class__.get_metrics(data['floor_mean'])
        metrics.loc['floor_mean'] = floor_mean

        # Calculate metrics for the ceiling mean values and store them in the DataFrame
        ceiling_mean = __class__.get_metrics(data['ceiling_mean'])
        metrics.loc['ceiling_mean'] = ceiling_mean

        # Calculate metrics for the average max values and store them in the DataFrame
        average_max = __class__.get_metrics(data['max'])
        metrics.loc['average_max'] = average_max

        # Calculate metrics for the floor max values and store them in the DataFrame
        floor_max = __class__.get_metrics(data['floor_max'])
        metrics.loc['floor_max'] = floor_max

        # Calculate metrics for the ceiling max values and store them in the DataFrame
        ceiling_max = __class__.get_metrics(data['ceiling_max'])
        metrics.loc['ceiling_max'] = ceiling_max

        # Calculate metrics for the average min values and store them in the DataFrame
        average_min = __class__.get_metrics(data['min'])
        metrics.loc['average_min'] = average_min

        # Calculate metrics for the floor min values and store them in the DataFrame
        floor_min = __class__.get_metrics(data['floor_min'])
        metrics.loc['floor_min'] = floor_min

        # Calculate metrics for the ceiling min values and store them in the DataFrame
        ceiling_min = __class__.get_metrics(data['ceiling_min'])
        metrics.loc['ceiling_min'] = ceiling_min

        path = dir_to_save + "/metrics_table.html"
        metrics.to_html(path)
        plots_paths.append(path)

        return plots_paths

    @staticmethod
    def get_metrics(data: pd.Series) -> dict: 
        """
        Calculate various metrics for a given data series.

        Parameters
        ----------
        data : pd.Series
            The data series for which metrics will be calculated.

        Returns
        -------
        dict
            A dictionary containing the calculated metrics: 
            MAD (Mean Absolute Deviation), 
            MAPD (Mean Absolute Percentage Deviation), 
            RMSD (Root Mean Square Deviation), 
            Emax (Maximum Absolute Deviation).
        """
        metrics = {}
        metrics['MAD'] = stats.mean_absolute_deviation(data, data.mean())
        metrics['MAPD'] = stats.mean_absolute_percentage_deviation(data, data.mean())
        metrics['RMSD'] = stats.root_mean_square_deviation(data, data.mean())
        metrics['Emax'] = stats.emax(data, data.mean())
        return metrics

    @staticmethod
    def confusion_matrix(data: pd.DataFrame, path_to_save:str="./plots") -> str:
        """
        Generate a confusion matrix plot based on the ground truth and predicted corner counts.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the ground truth and predicted corner counts.
        path_to_save : str, optional
            Path where the confusion matrix plot will be saved. Default is "./plots".

        Returns
        -------
        str
            The path to the saved confusion matrix plot.
        """
        unique_gt_corners = sorted(set(data['gt_corners']))
        unique_rb_corners = sorted(set(data['rb_corners']))

        unique_gt_corners.insert(0, 0)
        unique_rb_corners.insert(0, 0)

        gt_labels = [f"{unique_gt_corners[i]}-{unique_gt_corners[i+1]}" for i in range(len(unique_gt_corners) - 1)]
        rb_labels = [f"{unique_rb_corners[i]}-{unique_rb_corners[i+1]}" for i in range(len(unique_rb_corners) - 1)]
        
        gt_corners_class = pd.cut(data['gt_corners'], bins=unique_gt_corners, labels=gt_labels)
        rb_corners_class = pd.cut(data['rb_corners'], bins=unique_rb_corners, labels=rb_labels)

        cm = confusion_matrix(gt_corners_class, rb_corners_class)

        plt.figure(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=rb_labels, yticklabels=gt_labels)
        plt.xlabel('Predicted Corners')
        plt.ylabel('Actual Corners')
        plt.title('Confusion Matrix')

        plot_path = path_to_save + "/confusion_matrix.png"
        plt.savefig(plot_path)
        return plot_path 

    @staticmethod
    def box_plot(data: pd.DataFrame, label: str, title: str, path_to_save:str="./plots") -> str: 
        """
        Generate a box plot for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the data to be plotted.
        label : str
            Label for the data series in the plot.
        title : str
            Title of the plot.
        path_to_save : str, optional
            Path where the box plot will be saved. Default is "./plots".

        Returns
        -------
        str
            The path to the saved box plot.
        """
        melted = data.melt(var_name=label, value_name='deviation in degrees')

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=label, y='deviation in degrees', data=melted)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=20))

        plot_path = path_to_save + f"/box_plot_{label}.png"
        plt.savefig(plot_path)
        return plot_path 