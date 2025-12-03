from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

class SupervisedEnsemble:
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
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                loss = supervised_loss
                loss.backward()  # type: ignore
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)


    def test(self, step: int | None = None):
        
        for model in self.models:
            model.eval()

        test_losses = []

        with torch.no_grad():
            for x, targets in self.test_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                loss = torch.nn.functional.mse_loss(avg_preds, targets)
                test_losses.append(loss.item())

        test_loss = float(np.mean(test_losses))
        metrics = {"test_MSE": test_loss}

        self.logger.log_dict(metrics, step=step)

        return metrics


class CPSEnsemble:
    def __init__(
        self,
        supervised_criterion,
        unsupervised_weight,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device

        # If there is only one model, create anotherone from this
        if len(models) == 1:
            m1 = models[0]
            m2 = deepcopy(m1)

            # Add noise to the second model to have two different initial models
            for p in m2.parameters():
                if p.requires_grad:
                    p.data.add_(0.01 * torch.randn_like(p))

            self.models = [m1, m2]
        else:
            # In case there are two models
            self.models = models

        self.supervised_criterion = supervised_criterion
        self.unsupervised_weight = unsupervised_weight

        # Optim + scheduler for all the models
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Loaders
        self.sup_loader = datamodule.train_dataloader()
        self.unsup_loader = datamodule.unsupervised_train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        self.test_loader = datamodule.test_dataloader()

        # Logger (W&B)
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        with torch.no_grad():
            for x, targets in self.val_loader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(loss.item())

        return {"val_MSE": float(np.mean(val_losses))}

    def train(self, total_epochs, validation_interval):
        unsup_iter = iter(self.unsup_loader)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            sup_list = []
            unsup_list = []

            for x_sup, y_sup in self.sup_loader:
                x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)

                # Non-labelled Batch
                try:
                    x_unsup, _ = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(self.unsup_loader)
                    x_unsup, _ = next(unsup_iter)
                x_unsup = x_unsup.to(self.device)

                self.optimizer.zero_grad()

                # Supervised Loss
                sup_losses = [
                    self.supervised_criterion(m(x_sup), y_sup)
                    for m in self.models
                ]
                sup_loss = sum(sup_losses) / len(self.models)
                sup_list.append(sup_loss.item())

                # Cross Pseudo Supervision
                m1, m2 = self.models[0], self.models[1]

                p1 = m1(x_unsup)
                p2 = m2(x_unsup)

                loss_u1 = torch.nn.functional.mse_loss(p1, p2.detach())
                loss_u2 = torch.nn.functional.mse_loss(p2, p1.detach())
                cps_loss = 0.5 * (loss_u1 + loss_u2)

                unsup_list.append(cps_loss.item())

                # Total Loss
                loss = sup_loss + self.unsupervised_weight * cps_loss
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            sup_mean = float(np.mean(sup_list))
            unsup_mean = float(np.mean(unsup_list))

            summary = {
                "supervised_loss": sup_mean,
                "unsupervised_cps_loss": unsup_mean,
                "total_loss": sup_mean + self.unsupervised_weight * unsup_mean,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                summary.update(self.validate())
                pbar.set_postfix(summary)

            self.logger.log_dict(summary, step=epoch)

    
    def test(self, step: int | None = None):
       
        for model in self.models:
            model.eval()

        test_losses = []

        with torch.no_grad():
            for x, targets in self.test_loader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Prediction of the ensemble
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                loss = torch.nn.functional.mse_loss(avg_preds, targets)
                test_losses.append(loss.item())

        test_loss = float(np.mean(test_losses))
        metrics = {"test_MSE": test_loss}

        # Result to W&B
        self.logger.log_dict(metrics, step=step)

        return metrics