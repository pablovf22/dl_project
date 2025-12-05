from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
import os
import time
import pickle

def save_jp_history(history, method_name: str):
    """
    Save per-epoch training history to a file in jp_data/ so it can be
    loaded later from a Jupyter notebook to recreate plots.

    history: list of dicts, one dict per epoch (e.g. {"epoch": 1, "supervised_loss": ..., "val_MSE": ...})
    method_name: string describing the training method ("supervised", "cps", "mean_teacher", etc.)
    """
    os.makedirs("jp_data", exist_ok=True)
    timestamp = int(time.time())
    filename = os.path.join("jp_data", f"{method_name}_{timestamp}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(history, f)
    print(f"[jp_data] Saved training history to {filename}")

class SupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule
       ):
        self.device = device
        self.models = models
        self.jp_history = []

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
        # Reset history for this run
        self.jp_history = []

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
                # For consistency with semi-supervised trainers,
                # we define unsupervised_loss as 0 and total_loss = supervised_loss
                "unsupervised_loss": 0.0,
                "total_loss": supervised_losses_logged,
            }

            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            # Log to W&B
            self.logger.log_dict(summary_dict, step=epoch)

            # Save to in-memory history for Jupyter later
            epoch_record = {"epoch": epoch}
            epoch_record.update(summary_dict)
            self.jp_history.append(epoch_record)

        # When training finishes, dump history to jp_data/
        save_jp_history(self.jp_history, method_name="supervised_ensemble")



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
        self.jp_history = []

        # If there is only one model, create a second one from it (slightly perturbed)
        if len(models) == 1:
            m1 = models[0]
            m2 = deepcopy(m1)

            for p in m2.parameters():
                if p.requires_grad:
                    p.data.add_(0.01 * torch.randn_like(p))

            self.models = [m1, m2]
        else:
            self.models = models

        # Losses and weights
        self.supervised_criterion = supervised_criterion
        self.unsupervised_weight = unsupervised_weight

        # Optimizer and scheduler over all model parameters
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloaders: usa SIEMPRE estos nombres
        self.train_dataloader = datamodule.train_dataloader()
        self.unsupervised_train_dataloader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logger (W&B)
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(loss.item())

        return {"val_MSE": float(np.mean(val_losses))}

    def train(self, total_epochs, validation_interval):
        """
        Training loop for Cross Pseudo-Supervision (CPS).

        - Supervised loss on labeled data for both models.
        - Unsupervised CPS loss on unlabeled data:
          each model is trained to match the other's prediction.
        - At the end of training, a per-epoch history is saved to jp_data/
          for plotting in a Jupyter notebook.
        """
        self.jp_history = []

        sup_loader = self.train_dataloader
        unsup_loader = self.unsupervised_train_dataloader
        unsup_iter = iter(unsup_loader)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            supervised_losses = []
            unsupervised_losses = []

            for x_sup, y_sup in sup_loader:
                x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)

                # Get a batch of unlabeled data (cycle iterator if exhausted)
                try:
                    x_unsup, _ = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(unsup_loader)
                    x_unsup, _ = next(unsup_iter)
                x_unsup = x_unsup.to(self.device)

                self.optimizer.zero_grad()

                # ----- 1) SUPERVISED LOSS -----
                sup_losses_models = [
                    self.supervised_criterion(model(x_sup), y_sup)
                    for model in self.models
                ]
                sup_loss = sum(sup_losses_models)
                supervised_losses.append(sup_loss.item() / len(self.models))

                # ----- 2) CPS UNSUPERVISED LOSS -----
                preds_unsup = [model(x_unsup) for model in self.models]

                loss_0 = torch.nn.functional.mse_loss(
                    preds_unsup[0], preds_unsup[1].detach()
                )
                loss_1 = torch.nn.functional.mse_loss(
                    preds_unsup[1], preds_unsup[0].detach()
                )
                cps_loss = 0.5 * (loss_0 + loss_1)
                unsupervised_losses.append(cps_loss.item())

                # ----- 3) TOTAL LOSS -----
                loss = sup_loss + self.unsupervised_weight * cps_loss
                loss.backward()
                self.optimizer.step()

            # Scheduler step
            self.scheduler.step()

            sup_mean = float(np.mean(supervised_losses)) if supervised_losses else 0.0
            unsup_mean = float(np.mean(unsupervised_losses)) if unsupervised_losses else 0.0

            summary_dict = {
                "supervised_loss": sup_mean,
                "unsupervised_loss": unsup_mean,
                "total_loss": sup_mean + self.unsupervised_weight * unsup_mean,
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)

            epoch_record = {"epoch": epoch}
            epoch_record.update(summary_dict)
            self.jp_history.append(epoch_record)

        save_jp_history(self.jp_history, method_name="cps_ensemble")



class MeanTeacher:
    """
    Semi-supervised training using the Mean Teacher algorithm.

    - The student model is trained normally via gradient descent.
    - The teacher model is an exponential moving average (EMA) of the student.
    - L_total = L_supervised + lambda * L_consistency,
      where L_consistency encourages student predictions to match the teacher
      on unlabeled data.
    """

    def __init__(
        self,
        supervised_criterion,
        unsupervised_weight,
        ema_decay,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule
    ):
        self.device = device
        self.jp_history = []
        self.supervised_criterion = supervised_criterion
        self.unsupervised_weight = unsupervised_weight
        self.ema_decay = ema_decay
        self.logger = logger

        # Expect exactly one model: the student
        if len(models) != 1:
            raise RuntimeError(
                f"MeanTeacher expects exactly 1 model, but received {len(models)}."
            )

        self.student = models[0].to(self.device)

        # Create the teacher as a frozen EMA copy of the student
        self.teacher = deepcopy(self.student).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Optimizer and scheduler operate ONLY on the student
        self.optimizer = optimizer(params=list(self.student.parameters()))
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Data loaders: labeled, unlabeled, validation and test
        self.sup_loader = datamodule.train_dataloader()
        self.unsup_loader = datamodule.unsupervised_train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        self.test_loader = datamodule.test_dataloader()

    # ---------------------------------------------------------
    # EMA update for the teacher
    # ---------------------------------------------------------
    def _ema_update_teacher(self):
        """Update teacher parameters as an exponential moving average of the student."""
        with torch.no_grad():
            for p_teacher, p_student in zip(self.teacher.parameters(),
                                            self.student.parameters()):
                p_teacher.data.mul_(self.ema_decay).add_(
                    p_student.data, alpha=1.0 - self.ema_decay
                )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------
    def validate(self):
        """Evaluate validation MSE using the teacher model."""
        self.teacher.eval()

        losses = []
        with torch.no_grad():
            for x, targets in self.val_loader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = self.teacher(x)
                loss = torch.nn.functional.mse_loss(preds, targets)
                losses.append(loss.item())

        val_loss = float(np.mean(losses))
        return {"val_MSE": val_loss}

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------
    def test(self, step=None):
        """
        Evaluate test MSE using the teacher model.
        Logged to W&B as 'test_MSE'.
        """
        self.teacher.eval()

        losses = []
        with torch.no_grad():
            for x, targets in self.test_loader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = self.teacher(x)
                loss = torch.nn.functional.mse_loss(preds, targets)
                losses.append(loss.item())

        test_loss = float(np.mean(losses))
        metrics = {"test_MSE": test_loss}
        self.logger.log_dict(metrics, step=step)
        return metrics


    # ---------------------------------------------------------
    # TRAIN LOOP
    # ---------------------------------------------------------
    def train(self, total_epochs, validation_interval, rampup_epochs: int = 10):
        """
        Training loop for the Mean Teacher framework.

        - Supervised loss on labeled data (student only).
        - Consistency loss between student and teacher predictions on unlabeled data.
        - Teacher is updated as an exponential moving average (EMA) of the student.
        - The unsupervised loss weight is ramped up during the first `rampup_epochs`.
        - At the end of training, a per-epoch history is saved to jp_data/ for plotting.
        """
        # Reset history for this run
        self.jp_history = []

        sup_loader = self.sup_loader
        unsup_loader = self.unsup_loader
        unsup_iter = iter(unsup_loader)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.student.train()
            self.teacher.eval()  # teacher is only used for inference

            supervised_losses = []
            unsupervised_losses = []

            # Compute epoch-dependent weight for the unsupervised loss (ramp-up)
            if epoch <= rampup_epochs:
                lambda_u = self.unsupervised_weight * (epoch / rampup_epochs)
            else:
                lambda_u = self.unsupervised_weight

            for x_sup, y_sup in sup_loader:
                x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)

                # Fetch unlabeled batch (cycled if necessary)
                try:
                    x_unsup, _ = next(unsup_iter)
                except StopIteration:
                    unsup_iter = iter(unsup_loader)
                    x_unsup, _ = next(unsup_iter)
                x_unsup = x_unsup.to(self.device)

                self.optimizer.zero_grad()

                # ----- 1) SUPERVISED LOSS (student on labeled data) -----
                sup_preds = self.student(x_sup)
                sup_loss = self.supervised_criterion(sup_preds, y_sup)
                supervised_losses.append(sup_loss.item())

                # ----- 2) UNSUPERVISED CONSISTENCY LOSS (student vs teacher) -----
                with torch.no_grad():
                    teacher_preds_u = self.teacher(x_unsup)

                student_preds_u = self.student(x_unsup)
                unsup_loss = torch.nn.functional.mse_loss(
                    student_preds_u, teacher_preds_u
                )
                unsupervised_losses.append(unsup_loss.item())

                # ----- 3) TOTAL LOSS WITH RAMPED UNSUPERVISED WEIGHT -----
                loss = sup_loss + lambda_u * unsup_loss
                loss.backward()
                self.optimizer.step()

                # ----- 4) EMA UPDATE OF TEACHER -----
                self._ema_update_teacher()

            # Scheduler step per epoch
            self.scheduler.step()

            # Aggregate epoch stats
            sup_mean = float(np.mean(supervised_losses)) if supervised_losses else 0.0
            unsup_mean = float(np.mean(unsupervised_losses)) if unsupervised_losses else 0.0

            metrics = {
                "supervised_loss": sup_mean,
                "unsupervised_loss": unsup_mean,
                "total_loss": sup_mean + lambda_u * unsup_mean,
                "lambda_u": lambda_u,
            }

            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                metrics.update(val_metrics)
                pbar.set_postfix(metrics)

            # Log to W&B
            self.logger.log_dict(metrics, step=epoch)

            # Save epoch record for Jupyter
            epoch_record = {"epoch": epoch}
            epoch_record.update(metrics)
            self.jp_history.append(epoch_record)

        # Save full history to jp_data/ for notebook plotting
        save_jp_history(self.jp_history, method_name="mean_teacher_ensemble")