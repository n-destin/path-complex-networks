import torch
import numpy as np

from lib.helpers.criterion import get_criterion
from lib.helpers.evaluator import Evaluator
from lib.helpers.save_helpers import get_checkpoint_state, save_checkpoint, load_checkpoint
from lib.utils.log_utils import args_to_string
from lib.helpers.model_helpers import compute_params
from torch_geometric.loader import DataLoader as PyGDataLoader
from lib.datasets.graph_dataset_package.cospectral_graphs import build_pair_labels_from_graph
import networkx as nx

from tqdm import tqdm
import logging
import os
import wandb

from lib.data.complex import ComplexBatch


class Trainer(object):
    def __init__(
        self,
        model,
        args,
        train_loader,
        valid_loader,
        test_loader,
        train_ids,
        val_ids,
        test_ids,
        optimizer,
        scheduler,
        result_folder,
        train_seed,
        fold,
        device,
    ):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_seed = train_seed if train_seed is not None else 0
        self.fold = fold
        self.task_type = args.task_type
        self.device = device
        self.loss_fn = get_criterion(self.task_type)

        self.evaluator = Evaluator(self.args.eval_metric, eps=self.args.iso_eps)

        if not args.eval_only:
            self.best_val_epoch = 0
            self.valid_curve = []
            self.test_curve = []
            self.train_curve = []
            self.train_loss_curve = []
            self.params = []

            if not self.args.debug:
                new_folder = os.path.join(result_folder, f"seed_{self.train_seed}")
                self.filename = os.path.join(new_folder, f"results-seed_{self.train_seed}.txt")
                if fold is not None:
                    new_folder = os.path.join(new_folder, f"fold_{fold}")
                    self.filename = os.path.join(
                        new_folder, f"results-seed_{self.train_seed}-fold_{fold}.txt"
                    )
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                self.best_epoch_pt = os.path.join(new_folder, "best_weights.pt")

            print("==========================================================")
            print("Using device", str(device))
            print(f"Fold: {fold}")
            print(f"Random Seed: {args.seed}")
            print(f"Train Seed: {self.train_seed}")
            print("======================== Args ===========================")
            print(args)
            print("=========================================================")
            print(
                f"========================= training size: {len(self.train_loader.dataset)} ============================"
            )
            print(
                f"========================= testing size: {len(self.test_loader.dataset)} ============================"
            )
            print(
                f"========================= validation size: {len(self.valid_loader.dataset)} ============================"
            )

            compute_params(model)

    def load_saved_model(self, weights_file):
        load_checkpoint(self.model, self.optimizer, weights_file)

    def log_curves(self, log_dict, name, curves, suffix, is_best=False):
        prefix = "Best " if is_best else ""
        log_dict[f"{prefix}Train {name} {suffix}"] = curves[0]
        log_dict[f"{prefix}Val {name} {suffix}"] = curves[1]
        if len(curves) == 3:
            log_dict[f"{prefix}Test {name} {suffix}"] = curves[2]
        return log_dict

    def _num_samples_in_batch(self, batch):
        if isinstance(batch, ComplexBatch):
            num_samples = batch.cochains[0].x.size(0)
            for dim in range(1, batch.dimension + 1):
                num_samples = min(num_samples, batch.cochains[dim].num_cells)
            return num_samples
        return batch.x.size(0)

    def _normalize_split_ids(self, split_ids, task):
        if split_ids is None:
            return None

        if task == "pair":
            if torch.is_tensor(split_ids):
                split_ids = split_ids.tolist()
            return set(map(tuple, split_ids))

        if task == "node":
            if torch.is_tensor(split_ids):
                split_ids = split_ids.tolist()
            return set(split_ids)

        return None

    def _build_split_mask(self, batch, split_ids):
        """
        Returns a boolean mask aligned with targets/pred for node/pair tasks.
        Returns None for graph tasks or when split_ids is None.
        """
        if split_ids is None:
            return None

        if self.args.dataset == "cosc-structural-graphs":
            return None

        num_samples = self._num_samples_in_batch(batch)

        if self.args.task == "node":
            node_ids = self._normalize_split_ids(split_ids, "node")
            return torch.tensor(
                [u in node_ids for u in range(num_samples)],
                dtype=torch.bool,
                device=self.device,
            )

        if self.args.task == "pair":
            pair_ids = self._normalize_split_ids(split_ids, "pair")
            return torch.tensor(
                [(u, v) in pair_ids for u in range(num_samples) for v in range(u + 1, num_samples)],
                dtype=torch.bool,
                device=self.device,
            )

        return None

    def qtc_loss(self, batch: PyGDataLoader, pred: torch.Tensor) -> torch.Tensor:
        graphs = batch.to_data_list()
        labels = []
        Tmask = []

        train_node_ids = None
        train_pair_ids = None
        if self.args.task == "node":
            train_node_ids = self._normalize_split_ids(self.train_ids, "node")
        elif self.args.task == "pair":
            train_pair_ids = self._normalize_split_ids(self.train_ids, "pair")

        for graph in graphs:
            nx_graph = nx.Graph(graph.edge_list)
            labels.extend(build_pair_labels_from_graph(nx_graph))
            filtered_labels = []
            n = graph.num_nodes
            for u in range(n):
                for v in range(u + 1, n):
                    if self.args.task == "node" and (u in train_node_ids) and (v in train_node_ids):
                        filtered_labels.append(labels[u * n + v])
                    elif self.args.task == "pair" and ((u, v) in train_pair_ids):
                        filtered_labels.append(labels[u * n + v])

        labels = torch.tensor(labels, dtype=torch.float32, device=pred.device)

        mask = ~torch.isnan(labels)
        mask &= Tmask

        return self.loss_fn(pred[mask], labels[mask])

    def train_one_epoch(self):
        curve = []
        self.model.train()
        num_skips = 0

        for _, batch in enumerate(tqdm(self.train_loader, desc="Training iteration")):
            batch = batch.to(self.device)
            num_samples = self._num_samples_in_batch(batch)

            if num_samples <= 1:
                num_skips += 1
                if float(num_skips) / len(self.train_loader) >= 0.25:
                    logging.warning("Warning! 25% of the batches were skipped this epoch")
                continue

            if num_samples < 10:
                logging.warning(
                    "Warning! BatchNorm applied on a batch with only {} samples".format(num_samples)
                )

            add_constraint = (
                self.args.model in ["graphsage", "GAT", "GCN", "graphTransformer", "graphIsomorphism"]
                and self.args.dataset != "cosc-structural-graphs"
            )
            add_constraint = False # just for now

            self.optimizer.zero_grad()

            if add_constraint:
                pred, qt = self.model(batch, self.train_ids, self.val_ids, self.test_ids)
            else:
                pred = self.model(batch)

            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                targets = batch.y.argmax(dim=-1)
            else:
                targets = batch.y.to(torch.float32).view(pred.shape)

            mask = ~torch.isnan(targets)

            split_mask = self._build_split_mask(batch, self.train_ids)
            if split_mask is not None:
                mask = mask & split_mask

            if mask.sum() == 0:
                continue

            loss = self.loss_fn(pred[mask], targets[mask])

            if add_constraint:
                loss = loss + self.qtc_loss(batch, qt)

            loss.backward()
            self.optimizer.step()
            curve.append(loss.detach().cpu().item())

        epoch_train_loss = float(np.mean(curve)) if len(curve) else np.nan
        return epoch_train_loss

    def train(self):
        best_val_epoch = 0
        valid_curve = []
        test_curve = []
        train_curve = []

        suffix = f"(Seed {self.train_seed} - Fold {self.fold})"

        if not self.args.untrained:
            for epoch in range(1, self.args.epochs + 1):
                log_dict = {}

                print("=====Epoch {}".format(epoch))
                np.random.seed(np.random.get_state()[1][0] + epoch)

                print("Training...")
                epoch_train_loss = self.train_one_epoch()

                print("Evaluating...")
                if epoch == 1 or epoch % self.args.train_eval_period == 0:
                    train_perf, _ = self.eval(self.train_loader, self.train_ids)

                valid_perf, epoch_val_loss = self.eval(self.valid_loader, self.val_ids)

                if self.test_loader is not None:
                    test_perf, epoch_test_loss = self.eval(self.test_loader, self.test_ids)
                else:
                    test_perf = np.nan
                    epoch_test_loss = np.nan

                train_curve.append(train_perf)
                valid_curve.append(valid_perf)
                test_curve.append(test_perf)

                print(
                    f"Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}"
                    f" | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}"
                    f" | Test Loss {epoch_test_loss:.3f}"
                )

                if self.scheduler is not None:
                    if self.args.lr_scheduler == "ReduceLROnPlateau":
                        self.scheduler.step(valid_perf)
                        if self.args.early_stop and self.optimizer.param_groups[0]["lr"] < self.args.lr_scheduler_min:
                            print("\n!! The minimum learning rate has been reached.")
                            break
                    else:
                        self.scheduler.step()

                perf_curves = (
                    (train_perf, valid_perf, test_perf)
                    if not np.isnan(test_perf)
                    else (train_perf, valid_perf)
                )
                loss_curves = (
                    (epoch_train_loss, epoch_val_loss, epoch_test_loss)
                    if not np.isnan(epoch_test_loss)
                    else (epoch_train_loss, epoch_val_loss)
                )

                log_dict = self.log_curves(log_dict, "Performance", perf_curves, suffix, is_best=False)
                log_dict = self.log_curves(log_dict, "Loss", loss_curves, suffix, is_best=False)
                log_dict[f"Epoch {suffix}"] = epoch

                if (
                    (valid_perf <= valid_curve[best_val_epoch] and self.args.minimize)
                    or (valid_perf >= valid_curve[best_val_epoch] and not self.args.minimize)
                ):
                    best_val_epoch = epoch - 1
                    log_dict = self.log_curves(
                        log_dict, "Performance", perf_curves, suffix, is_best=True
                    )
                    if not self.args.debug:
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, epoch),
                            self.best_epoch_pt,
                        )

                if not self.args.debug:
                    wandb.log(log_dict)
        else:
            train_curve.append(np.nan)
            valid_curve.append(np.nan)
            test_curve.append(np.nan)

        print("Final Evaluation...")
        final_train_perf = np.nan
        final_val_perf = np.nan
        final_test_perf = np.nan

        if not self.args.dataset.startswith("sr"):
            final_train_perf, _ = self.eval(self.train_loader, self.train_ids)
            final_val_perf, _ = self.eval(self.valid_loader, self.val_ids)

        if self.test_loader is not None:
            final_test_perf, _ = self.eval(self.test_loader, self.test_ids)

        msg = (
            f"========== Result ============\n"
            f"Dataset:        {self.args.dataset}\n"
            f"------------ Best epoch -----------\n"
            f"Train:          {train_curve[best_val_epoch]}\n"
            f"Validation:     {valid_curve[best_val_epoch]}\n"
            f"Test:           {test_curve[best_val_epoch]}\n"
            f"Best epoch:     {best_val_epoch}\n"
            "------------ Last epoch -----------\n"
            f"Train:          {final_train_perf}\n"
            f"Validation:     {final_val_perf}\n"
            f"Test:           {final_test_perf}\n"
            "-------------------------------\n\n"
        )
        print(msg)

        msg += args_to_string(self.args)

        curves = {
            "train": train_curve,
            "val": valid_curve,
            "test": test_curve,
            "last_val": final_val_perf,
            "last_test": final_test_perf,
            "last_train": final_train_perf,
            "best": best_val_epoch,
        }

        if not self.args.debug:
            with open(self.filename, "w") as handle:
                handle.write(msg)
            wandb.save(self.filename)

        return curves

    def eval(self, loader, split_ids=None, return_perf_each_sample=False):
        """
        Evaluates a model over all batches in a loader.
        Masks both loss and metric to the requested split.
        """
        loss_fn = get_criterion(self.task_type)
        self.model.eval()

        y_true = []
        y_pred = []
        losses = []

        for _, batch in enumerate(tqdm(loader, desc="Eval iteration")):
            if torch.get_default_dtype() == torch.float64 and isinstance(batch, ComplexBatch):
                for dim in range(batch.dimension + 1):
                    batch.cochains[dim].x = batch.cochains[dim].x.double()
                    assert batch.cochains[dim].x.dtype == torch.float64, batch.cochains[dim].x.dtype

            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch)

                if self.task_type == "isomorphism":
                    assert self.loss_fn is None
                    continue

                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    targets = batch.y.argmax(dim=-1)
                else:
                    targets = batch.y.to(torch.float32).view(pred.shape)

                mask = ~torch.isnan(targets)

                split_mask = self._build_split_mask(batch, split_ids)
                if split_mask is not None:
                    mask = mask & split_mask

                if mask.sum() == 0:
                    continue

                masked_pred = pred[mask]
                masked_targets = targets[mask]

                loss = loss_fn(masked_pred, masked_targets)
                losses.append(loss.detach().cpu().item())

                y_pred.append(masked_pred.detach().cpu())
                y_true.append(masked_targets.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy() if len(y_true) else None
        y_pred = torch.cat(y_pred, dim=0).numpy() if len(y_pred) else None

        input_dict = {"y_pred": y_pred, "y_true": y_true}
        mean_loss = float(np.mean(losses)) if len(losses) else np.nan

        if return_perf_each_sample:
            return self.evaluator.eval_each_sample(input_dict), self.evaluator.eval(input_dict), mean_loss
        return self.evaluator.eval(input_dict), mean_loss