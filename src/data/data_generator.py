import torch
import numpy as np
import h5py
import os
import gc
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split

class BaseDataGenerator(ABC):
    def __init__(self, config):
        self.config = config
        self.n_samples = config["data"]["n_samples"]
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    @abstractmethod
    def generate_data(self):
        pass

    def prepare_data(self):
        inputs, outputs = self.generate_data()

        test_size = int(self.config["training"]["test_split"] * self.n_samples)
        val_size = int(self.config["training"]["validation_split"] * self.n_samples)
        train_size = self.n_samples - test_size - val_size

        dataset = TensorDataset(inputs, outputs)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Scale data
        train_inputs, train_outputs = train_dataset[:]
        val_inputs, val_outputs = val_dataset[:]
        test_inputs, test_outputs = test_dataset[:]

        # Fit scalers on training data
        train_inputs_scaled = torch.from_numpy(
            self.input_scaler.fit_transform(train_inputs)
        ).float()
        train_outputs_scaled = torch.from_numpy(
            self.output_scaler.fit_transform(train_outputs)
        ).float()

        # Transform validation and test data
        val_inputs_scaled = torch.from_numpy(
            self.input_scaler.transform(val_inputs)
        ).float()
        val_outputs_scaled = torch.from_numpy(
            self.output_scaler.transform(val_outputs)
        ).float()
        test_inputs_scaled = torch.from_numpy(
            self.input_scaler.transform(test_inputs)
        ).float()
        test_outputs_scaled = torch.from_numpy(
            self.output_scaler.transform(test_outputs)
        ).float()

        return (
            (train_inputs_scaled, train_outputs_scaled),
            (val_inputs_scaled, val_outputs_scaled),
            (test_inputs_scaled, test_outputs_scaled)
        )

    def get_data_loaders(self):
        (train_inputs, train_outputs), \
        (val_inputs, val_outputs), \
        (test_inputs, test_outputs) = self.prepare_data()

        train_dataset = TensorDataset(train_inputs, train_outputs)
        val_dataset = TensorDataset(val_inputs, val_outputs)
        test_dataset = TensorDataset(test_inputs, test_outputs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False
        )

        return train_loader, val_loader, test_loader

class HybridPiecewiseDataGenerator(BaseDataGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.gamma_th = config["data"]["gamma_th"]
        self.K = config["data"]["K"]
        self.Gamma = config["data"]["Gamma"]
        self.rho_breaks = config["data"]["rho_breaks"]

    def generate_data(self):
        # Generate density and velocity samples
        rho = self.config["data"]["rho_min"] + \
              (self.config["data"]["rho_max"] - self.config["data"]["rho_min"]) * \
              torch.rand(self.n_samples, 1)
        vx = self.config["data"]["vx_max"] * torch.rand(self.n_samples, 1)
        W = (1 - vx**2).pow(-0.5)

        # Calculate polytropic parameters
        a_i = torch.zeros(len(self.K))
        K_values = self.K.copy()

        for i in range(1, len(self.K)):
            K_values[i] = K_values[i-1] * self.rho_breaks[i-1]**(self.Gamma[i-1] - self.Gamma[i])
            a_i[i] = a_i[i-1] + K_values[i-1] * self.rho_breaks[i-1]**(self.Gamma[i-1] - 1) / \
                     (self.Gamma[i-1] - 1) - K_values[i] * self.rho_breaks[i-1]**(self.Gamma[i] - 1) / \
                     (self.Gamma[i] - 1)

        # Calculate pressure and energy density
        p = torch.zeros_like(rho)
        eps_cold = torch.zeros_like(rho)

        for i in range(len(self.K)):
            if i == 0:
                mask = rho < self.rho_breaks[i]
            elif i < len(self.K) - 1:
                mask = (rho >= self.rho_breaks[i-1]) & (rho < self.rho_breaks[i])
            else:
                mask = rho >= self.rho_breaks[i-1]

            eps_cold[mask] = a_i[i] + K_values[i] * rho[mask]**(self.Gamma[i] - 1) / (self.Gamma[i] - 1)
            p[mask] = K_values[i] * rho[mask]**self.Gamma[i]

        # Add thermal component
        eps_th = self.config["data"]["eps_min"] + \
                 (self.config["data"]["eps_max"] - self.config["data"]["eps_min"]) * \
                 torch.rand(self.n_samples, 1)
        p_th = (self.gamma_th - 1) * rho * eps_th
        p += p_th

        h = 1 + eps_cold + eps_th + p/rho

        # Calculate conserved variables
        D = rho * W
        Sx = rho * h * W**2 * vx
        tau = rho * h * W**2 - p - D

        inputs = torch.cat((D, Sx, tau), dim=1)
        outputs = p

        return inputs, outputs

    def prepare_data(self):
        inputs, outputs = self.generate_data()

        # Split data
        test_size = int(self.config["training"]["test_split"] * self.n_samples)
        val_size = int(self.config["training"]["validation_split"] * self.n_samples)
        train_size = self.n_samples - test_size - val_size

        # Create datasets
        dataset = TensorDataset(inputs, outputs)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Scale data
        train_inputs, train_outputs = train_dataset[:]
        val_inputs, val_outputs = val_dataset[:]
        test_inputs, test_outputs = test_dataset[:]

        # Fit scalers on training data
        train_inputs_scaled = torch.from_numpy(
            self.input_scaler.fit_transform(train_inputs)
        ).float()
        train_outputs_scaled = torch.from_numpy(
            self.output_scaler.fit_transform(train_outputs)
        ).float()

        # Transform validation and test data
        val_inputs_scaled = torch.from_numpy(
            self.input_scaler.transform(val_inputs)
        ).float()
        val_outputs_scaled = torch.from_numpy(
            self.output_scaler.transform(val_outputs)
        ).float()
        test_inputs_scaled = torch.from_numpy(
            self.input_scaler.transform(test_inputs)
        ).float()
        test_outputs_scaled = torch.from_numpy(
            self.output_scaler.transform(test_outputs)
        ).float()

        return (
            (train_inputs_scaled, train_outputs_scaled),
            (val_inputs_scaled, val_outputs_scaled),
            (test_inputs_scaled, test_outputs_scaled)
        )

    def get_data_loaders(self):
        (train_inputs, train_outputs), \
        (val_inputs, val_outputs), \
        (test_inputs, test_outputs) = self.prepare_data()

        train_dataset = TensorDataset(train_inputs, train_outputs)
        val_dataset = TensorDataset(val_inputs, val_outputs)
        test_dataset = TensorDataset(test_inputs, test_outputs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False
        )

        return train_loader, val_loader, test_loader

class TabulatedDataGenerator(BaseDataGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.eos_tables_dir = config["paths"]["eos_tables"]
        self.eos_table_filename = config["data"]["eos_table_filename"]

    @staticmethod
    def cleanup_h5py():
        for obj in gc.get_objects():
            if isinstance(obj, h5py.File):
                try:
                    obj.close()
                except:
                    pass
        gc.collect()

    @staticmethod
    def W(v):
        v_sqr = v ** 2 if isinstance(v, float) else np.sum(v ** 2)
        return (1 - v_sqr) ** (-1 / 2)

    def generate_data(self):
        try:
            self.cleanup_h5py()
            eos_file = os.path.join(self.eos_tables_dir, self.eos_table_filename)
            with h5py.File(eos_file, 'r') as eos_table:
                ye_table = eos_table["ye"][()]
                temp_table = eos_table["logtemp"][()]
                rho_table = eos_table["logrho"][()]
                eps_table = eos_table["logenergy"][()]
                p_table = eos_table["logpress"][()]

                len_ye = eos_table["pointsye"][()][0]
                len_temp = eos_table["pointstemp"][()][0]
                len_rho = eos_table["pointsrho"][()][0]

                features, labels = [], []
                V_MIN, V_MAX = 0, 0.721

                for _ in range(self.n_samples):
                    v = np.random.uniform(V_MIN, V_MAX)
                    ye_index = np.random.choice(len_ye)
                    temp_index = np.random.choice(len_temp)
                    rho_index = np.random.choice(len_rho)

                    ye = ye_table[ye_index]
                    logtemp = temp_table[temp_index]
                    logrho = rho_table[rho_index]
                    logeps = eps_table[ye_index, temp_index, rho_index]
                    logp = p_table[ye_index, temp_index, rho_index]

                    rho = 10 ** logrho
                    eps = 10 ** logeps
                    p = 10 ** logp

                    h = 1 + eps + p / rho
                    w = self.W(v)
                    D = rho * w
                    S = rho * h * w ** 2 * v
                    tau = rho * h * w ** 2 - p - D

                    features.append([np.log10(D), np.log10(S), np.log10(tau), ye])
                    labels.append([logp])

            return torch.tensor(features, dtype=torch.float32), \
                   torch.tensor(labels, dtype=torch.float32)

        finally:
            self.cleanup_h5py()
