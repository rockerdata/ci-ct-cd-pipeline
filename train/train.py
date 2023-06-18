import wandb
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle as pkl
from sklearn import svm
from sklearn import datasets
from config.config import CONFIG

defaults=dict( dropout = 0.2,
                    hidden_layer_size = 128,
                    layer_1_size = 16,
                    layer_2_size = 32,
                    learn_rate = 0.01,
                    decay = 1e-6,
                    momentum = 0.9,
                    epochs = 5,
                    )

        # initialize a new wandb run
wandb.login(key=os.environ['WANDB_API_KEY'])
wandb.init(project=CONFIG['project_name'], config=defaults)
config = wandb.config

class Train:
    def __init__(self) -> None:
        pass


    def preprocess(self):
        wbcd = wisconsin_breast_cancer_data = datasets.load_breast_cancer()
        self.feature_names = wbcd.feature_names
        self.labels = wbcd.target_names
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(wbcd.data, wbcd.target, test_size = 0.2, random_state = 42)

    def train(self):
        self.model = RandomForestClassifier()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_probas = self.model.predict_proba(self.X_test)
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        
        with open(os.path.join(wandb.run.dir, "model.pkl"),'wb')as f:
            pkl.dump(self.model,f)
        artifact = wandb.Artifact(name='github_wandb', type='model')
        artifact.add_file(
                        local_path=os.path.join(wandb.run.dir, "model.pkl"), 
                        name='model.pkl'
                        )
        wandb.log_artifact(artifact)

        # Save a model file manually from the current directory:
        wandb.save('model.pkl')

    
    def print_metrics(self):
        self.y_pred = self.model.predict(self.X_test)
        self.y_probas = self.model.predict_proba(self.X_test)
        print(metrics.classification_report(self.y_test, self.y_pred))
        print("roc_auc_score: ", metrics.roc_auc_score(self.y_test, self.y_pred))
        print("f1 score: ", metrics.f1_score(self.y_test, self.y_pred))

    def wandb_plot_metrics(self):
        wandb.sklearn.plot_classifier(self.model, 
                              self.X_train, self.X_test, 
                              self.y_train, self.y_test, 
                              self.y_pred, self.y_probas,
                              self.labels,
                              is_binary=True,
                              model_name='Random Forest Classifier')
        wandb.finish()

    def train_model(self):
        self.preprocess()
        self.train()
        self.wandb_plot_metrics()
        

# if __name__=="__main__":
#     train_obj = Train()
#     train_obj.train_model()

