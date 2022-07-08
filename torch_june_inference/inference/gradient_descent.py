import torch
import pandas as pd
import pyro
from torch_june_inference.inference.base import InferenceEngine


class GradientDescent(InferenceEngine):
    def _get_optimizer(self):
        config = self.inference_configuration["optimizer"]
        optimizer_class = getattr(torch.optim, config.pop("type"))
        optimizer = optimizer_class(self.runner.model.parameters(), **config)
        return optimizer

    def _get_optimizer_and_scheduler(self):
        config = self.inference_configuration["optimizer"]
        optimizer_type = config.pop("type")
        if "milestones" in config:
            milestones = config.pop("milestones")
        else:
            milestones = []
        if "gamma" in config:
            gamma = config.pop("gamma")
        else:
            gamma = 1.0
        optimizer_class = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_class(self.runner.model.parameters(), **config)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        return optimizer, scheduler

    def _get_loss(self):
        config = self.inference_configuration["loss"]
        loss_class = getattr(torch.nn, config.pop("type"))
        loss = loss_class(**config)
        return loss

    def run(self, verbose=True):
        names_to_save = self._set_initial_parameters()
        optimizer, scheduler = self._get_optimizer_and_scheduler()
        loss_fn = self._get_loss()
        df = pd.DataFrame()
        n_epochs = self.inference_configuration["n_epochs"]
        y_obs = self.observed_data
        for epoch in range(n_epochs):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y = self.runner()
            loss = torch.zeros(1, requires_grad=True, device=self.device)
            for key in self.data_observable:
                time_stamps = self.data_observable[key]["time_stamps"]
                if time_stamps == "all":
                    time_stamps = range(len(y[key]))
                data = y[key][time_stamps]
                data_obs = y_obs[key][time_stamps]
                loss = loss + loss_fn(data, data_obs)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.runner.model.parameters(), 1e-2)
            print("---")
            print(self.runner.model.infection_networks.networks.household.log_beta.grad)
            print(self.runner.model.infection_networks.networks.school.log_beta.grad)
            optimizer.step()
            running_loss = loss.item()
            if verbose:
                print(f"[{epoch + 1}] loss: {running_loss:.10e}")
            for name, param in self.runner.model.named_parameters():
                if name in names_to_save:
                    df.loc[epoch, name] = param.item()
            df.loc[epoch, "loss"] = running_loss
            df.to_csv(self.results_path / "training.csv", index=False)
            scheduler.step()
        model_to_save_path = self.results_path / "model.path"
        torch.save(self.runner.model.state_dict(), model_to_save_path)
