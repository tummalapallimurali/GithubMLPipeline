import matplotlib.pyplot as plt 
import pandas as pd 
import skops.io as sko
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_recall_fscore_support

from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, r2_score, roc_auc_score, roc_curve, accuracy_score , confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression

class ChurnModel:
    def __init__(self, path, target_variable, index=None, random_state= 50):
        self.path = path
        self.target_variable= target_variable
        self.index = index 
        self.random_state = random_state
        self.model_pipeline = None 
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.preprocessor = None

    def load_dataset(self, drop_columns, n_rows=None):
        print("loading dataset...")
        df = pd.read_csv(self.path,index_col= self.index, nrows = n_rows)
        if drop_columns:
            df = df.drop(drop_columns,axis = 1)
        df = df.sample(frac=1)
        X = df.drop(self.target_variable, axis =1) 
        y = df[self.target_variable]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y , test_size= 0.3, random_state= self.random_state)
        print("data split completed successfully")

    def data_preprocesing(self, cat_cals, num_cols):
        print("data preprocessing....")
        numerical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), 
                    ("scalar" , StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[("imputer" , SimpleImputer(strategy="most_frequent")),
                     ("scalar" , OrdinalEncoder())]
        )
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num" , numerical_transformer, num_cols),
                ("cat", categorical_transformer, cat_cals)
            ])
        print("finsihed data transformation")

    def build_model(self, k_best = 5):
        print("buidlign pipeline")
        feature_selector = SelectFromModel(LogisticRegression(max_iter=1000))
        
        model = GradientBoostingClassifier(n_estimators=100, random_state= self.random_state)
        
        train_pipeline = Pipeline(
            steps=[("feature_selection", feature_selector),
                   ("GModel", model)]
        )        

        self.model_pipeline = Pipeline(
            steps = [("preprocessor", self.preprocessor),
            ("train", train_pipeline)]
        )
        print("model pipeline built")

    def train_model(self):
        if self.model_pipeline is None:
            raise ValueError("Error")
        print("training")
        self.model_pipeline.fit(self.X_train,self.y_train)
        print("done training")
    
    def evaluate_model(self):
        print("evaluation")
        predictions = self.model_pipeline.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average = "macro")
        print("Accuracy", accuracy)
        print("F1 score", f1)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, predictions, average = "macro")
        print("Precision", precision)
        print("Recall", recall)
        return accuracy, f1
    
    def plot_confusion_matrix(self):
        predictions = self.model_pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test , predictions, labels = self.model_pipeline.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels= self.model_pipeline.classes_)
        disp.plot()
        plt.savefig("result.png")
        print("Confustion matrix saved")

    def roc_curve(self):
        predictions = self.model_pipeline.predict(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, predictions)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig("roc_curve.png")
        print("ROC curve saved")


    def save_metrics(self, accuracy,f1):
        with open("metrics.txt", "w") as outfile:
            outfile.write(f"Accuracy= {accuracy}, f1score ={f1_score}")
        print("metrics saved")
    
    def save_pipeline(self):
        print('Saving pipeline')
        sko.dump(self.model_pipeline, "Churnpipeline.skops")
        print("pipeline saved")

if __name__ == "__main__":
    # config
    data_file = 'Churn_Modelling.csv'
    target_col = "Exited"
    drop_cols = ["RowNumber","CustomerId","Surname"]
    cat_cols = [1,2]
    num_cols = [0,3,4,5,6,7,8,9]

    # initialize 
    ChurnMod = ChurnModel(data_file, target_col)
    ChurnMod.load_dataset(drop_cols, n_rows = 1000)
    ChurnMod.data_preprocesing(cat_cols, num_cols)
    ChurnMod.build_model()
    # training
    ChurnMod.train_model()
    accuracy_score, f1 = ChurnMod.evaluate_model()

    # plotting

    ChurnMod.plot_confusion_matrix()
    ChurnMod.roc_curve()
    #saving
    ChurnMod.save_metrics(accuracy_score, f1)

    # save pipline 
    #ChurnMod.save_pipeline()




        

