from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from torch.nn import functional as F

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.unsup_dataloader = datamodule.unsupervised_train_dataloader() #Load UNsupervised data
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        unsup_iterator = iter(self.unsup_dataloader)
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            cps_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                try:
                    unsup_batch = next(unsup_iterator)
                except StopIteration:
                    unsup_iterator = iter(self.unsup_dataloader)
                    unsup_batch = next(unsup_iterator)


                #print("DEBUG unsup_batch type:", type(unsup_batch))
                #print("DEBUG first element type:", type(unsup_batch[0]))
                #print("DEBUG batch element keys:", 
                 #       unsup_batch[0].__dict__.keys() if hasattr(unsup_batch[0], "__dict__") else "NO ATTRIBUTES")
                if isinstance(unsup_batch, list):
                    unsup_batch = unsup_batch[0].to(self.device)
                else:
                    unsup_batch = unsup_batch.to(self.device)

                self.optimizer.zero_grad()

                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)


                # CPS loss: model A uses model B's pseudo-labels and vice versa
                pred_A = self.models[0](unsup_batch)
                pred_B = self.models[1](unsup_batch)

                pseudo_labels_B = torch.softmax(pred_B.detach(), dim=-1) #Softmax for probability distribution
                pseudo_labels_A = torch.softmax(pred_A.detach(), dim=-1)

                cps_loss_A = F.cross_entropy(pred_A, pseudo_labels_B.argmax(dim=-1)) #Cross entropy with pseudo-labels (Pred A with Pseudo-labels from B)
                cps_loss_B = F.cross_entropy(pred_B, pseudo_labels_A.argmax(dim=-1))

                cps_loss = (cps_loss_A + cps_loss_B) / 2 #Compute avg CPS loss

                loss = supervised_loss + cps_loss #Compute total loss
                loss.backward() #Backpropagate
                self.optimizer.step()

                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                cps_losses_logged.append(cps_loss.detach().item())
            
            self.scheduler.step()
            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "cps_loss": cps_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
                
            self.logger.log_dict(summary_dict, step=epoch)
