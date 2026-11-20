from pathlib import Path
import copy
import time
from datetime import datetime
import json

import numpy as np
import pandas as pd

import torch
#from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.lr_scheduler import StepLR, MultiStepLR



class trainer:
    def __init__(self,
        model,
        criterion,
        train_loader,
        test_loader,
        epochs,
        device,
        info=None
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.002, momentum=0.9)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.2)
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[5, 10, 15], gamma=0.2)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs 
        self.info = info if info is not None else {}
        self.training_report = []
        self.test_losses = []
        self.test_predictions = None
        self.best_loss = float('inf')
        self.best_state = None
        self.timestamp = None   # data and time of the end of training

    def _early_stopper(self, n):
        # if after n last epochs test loss increased: stop training
        stop = False
        if len(self.test_losses) >= n + 1:
            last_n = self.test_losses[-(n+1):]
            if all(np.diff(last_n) > 0):
                stop = True
        return stop

    def train(self):
        start_time = time.time()
        for epo in range(1, self.epochs + 1):

            self.model.train()
            
            counter = 0
            running_loss = 0
            
            for batch in self.train_loader:
                x = [t.to(self.device) for t in batch[:-1]]
                y = batch[-1].to(self.device)
                
                loss = self.criterion(self.model(*x), y)
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                counter += 1
                running_loss += loss.item()
        
                if counter % 1000 == 0:
                    report_string = f"Epoch: {epo}, batch: {counter}, loss: {running_loss/counter}"
                    self.training_report.append(report_string)
                    print(report_string)
        
            # validation
            self.model.eval()
            
            test_loss = 0
        
            with torch.no_grad():
                for batch in self.test_loader:
                    x = [t.to(self.device) for t in batch[:-1]]
                    y = batch[-1].to(self.device)
                    
                    t_loss = self.criterion(self.model(*x), y)
                    test_loss += t_loss.item()
                    
            test_loss_norm = test_loss / len(self.test_loader)
            report_string_test = f"Loss on test set: {test_loss_norm}\n"
            self.test_losses.append(test_loss_norm)
            self.training_report.append(report_string_test)
            print(report_string_test)

            if test_loss_norm < self.best_loss:
                self.best_loss = test_loss_norm
                self.best_state = copy.deepcopy(
                    self.model.state_dict()
                )

            if self._early_stopper(3):
                self.info.update({"training_time": time.time() - start_time})
                self.timestamp = datetime.now().strftime("%d%m%Y_%H_%M_%S")
                break

            if self.scheduler is not None:
                self.scheduler.step()

        self.info.update({"training_time": time.time() - start_time})
        self.timestamp = datetime.now().strftime("%d%m%Y_%H_%M_%S")


    def print_report(self):
        print("\n".join(self.training_report))


    def predict_test(self, label_mapping, with_prob=True, sep_nts=False, ds_out=False):
        # label_mapping must be index-to-label mapping.
        # For separated nucleotides in output (sep_nts=True)
        # it must be index to base mapping e.g. {0: "A", 1: "G", ...}.
        # For joined nucleotides in output (sep_nts=False)
        # it must be index to two-nts-step mapping e.g. {0: "AA", 1: "AC", ...}
        # Mapping must be compatible with the one used for y encoding.
        
        self.model.eval()
    
        if sep_nts:
            # 2 X 4 classes; nucleotides in steps encoded separately; 
            # two predictions -- one for each position in the step
            target_cols = ["real1", "real2", "pred1", "pred2"]
            prob_cols = list(label_mapping.values()) * 2
    
        else:
            # 16 classes; steps encoded jointly; single predicted value
            target_cols = ["real", "pred"]
            prob_cols = list(label_mapping.values())
    
        col_labels = target_cols + (prob_cols if with_prob else [])
        
        predictions = []
    
        with torch.no_grad():
            for batch in self.test_loader:
                x = [t.to(self.device) for t in batch[:-1]]
                y = batch[-1].to(self.device)
                
                model_out = self.model(*x)

                if ds_out:
                    # Average probabilities from both strands
                    mean_dim = -3 if sep_nts else -2
                    model_out = torch.softmax(model_out, dim=-1).mean(dim=mean_dim)
                    
                pred = torch.argmax(model_out, dim=-1)
    
                values = [y.reshape(y.shape[0], -1),
                          pred.reshape(pred.shape[0], -1)]
    
                if with_prob:
                    prob = model_out if ds_out else torch.softmax(model_out, dim=-1)
                    values.append(prob.reshape(prob.shape[0], -1))
    
                predictions.append(torch.cat(values, dim=1).cpu())

        pred_all = torch.cat(predictions).numpy()
        pred_df = pd.DataFrame(pred_all, columns=col_labels)
    
        for col in target_cols:
            pred_df[col] = pred_df[col].astype(int).map(label_mapping)
    
        if sep_nts:
            pred_df["real"] = pred_df["real1"] + pred_df["real2"]
            pred_df["pred"] = pred_df["pred1"] + pred_df["pred2"]
            pred_df.drop(columns=target_cols, inplace=True)
    
            for col in ["pred", "real"]:
                pred_df.insert(0, col, pred_df.pop(col))
    
        self.test_predictions = pred_df
    
        return pred_df


    def test_accuracy(self):
        if self.test_predictions is not None:
            n_all = len(self.test_predictions)
            n_good = (self.test_predictions["real"] == self.test_predictions["pred"]).sum()

            self.info.update({"test_accuracy": n_good / n_all})
            
            return n_good / n_all
        else:
            print("Run .predict_test() first.")

    
    def example_out(self):
        self.model.eval()

        exmpl_x = next(iter(self.train_loader))[:-1]
        exmpl_x = [t.to(self.device) for t in exmpl_x]
        
        with torch.no_grad():
            return self.model(*exmpl_x)
    

    def restore_best_model(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        else:
            print("No model state. Train model first.")

    
    def save_model(self, path, with_info=True):

        state_file = f"{Path(path).stem}_{self.timestamp}.pt"
        state_path = Path(path).parent / state_file
        torch.save(self.model.state_dict(), state_path)

        if with_info:
            if self.info is not None:
                #with_seq_str = "seq" if self.info.get("with_seq") else ""
                #file_suffix = "_".join(
                #    self.info.get(i, "") 
                #    if isinstance(self.info.get(i, ""), str) 
                #    else with_seq_str
                #    for i in ["model", "model_family", "with_seq", "y_form"]
                #)
                #info_file = f"{Path(path).stem}_{file_suffix}_{timestamp}.json"
                
                info_file = f"{Path(path).stem}_{self.timestamp}.json"
                info_path = Path(path).parent / info_file
                with open(info_path, "w") as f:
                    json.dump(self.info, f, indent=4)
            else:
                print("No info. Info file won't be saved.")

    
    def load_model(self, path, info_path=None):
        """
        Loads model weights saved to .pt file. 
        Optionally loads model info from .json file.

        input: 
        path: path to .pt file with weights
        info_path: path to .json info file
        """
        self.model.load_state_dict(
            torch.load(
                path, 
                map_location=self.device,
                weights_only=True
            )
        )

        if info_path is not None:
            with open(info_path, "r") as f:
                self.info = json.load(f)