import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from util import create_info_file, init_log, load_yaml
from cnn import metrics, model  # Assuming ModelMetrics is here
from sklearn.metrics import confusion_matrix
from medmnist import Evaluator

class BaseTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, test_loader,
                params: dict, logger=None):
        self.model = model.to(params['device'])
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.params = params
        self.device = torch.device(params['device'])
        self.logger = logger or init_log("INFO", name=__name__)
        self.scaler = GradScaler(enabled=self.params.get('mixed_precision', False))
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.early_stop_patience = params.get('early_stop_patience', 10)
        self.no_improvement_epochs = 0
        self.best_model_path = os.path.join(self.params['model_path'], 'best_model.pth')
        if not os.path.exists(self.params['model_path']):
            os.makedirs(self.params['model_path'])
        # Optionally, a checkpoint directory
        self.checkpoint_dir = os.path.join(self.params['model_path'], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.params.get('mixed_precision', False)):
                outputs = self.model(inputs)
                if self.params.get('task', 'classification') == 'multi-class':
                    labels = labels.squeeze().long()
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.params.get('mixed_precision', False)):
                    outputs = self.model(inputs)
                    if self.params.get('task', 'classification') == 'multi-class':
                        labels = labels.squeeze().long()
                    loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        avg_loss = epoch_loss / len(loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def compute_additional_metrics(self, loader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        y_score = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                if self.params.get('task', 'classification') == 'multi-class':
                    y_score = torch.cat((y_score, outputs.softmax(dim=-1)), 0)
        if self.params.get('task', 'classification') == 'multi-class':
            y_score = y_score.cpu().detach().numpy()
            evaluator = Evaluator(self.params['dataset'], split='test', root=self.params['data_path'])
            auc, acc = evaluator.evaluate(y_score)
        else:
            auc, acc = 0, 0
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        return conf_matrix, auc, acc

    def update_scheduler(self, scheduler, metric=None):
        if scheduler is not None:
            if self.params.get('lr_scheduler') == 'reduce_on_plateau' and metric is not None:
                scheduler.step(metric)
            else:
                scheduler.step()

    def release_gpu_memory(self):
        if not torch.cuda.is_available():
            self.logger.info("CUDA is not available. No GPU memory to release.")
            return
        torch.cuda.empty_cache()
        self.logger.info("GPU cache cleared.")

    # def reset_and_load_best_model(self, best_model_path):
    #     # This method should be overridden in subclasses if special logic is needed.
    #     self.model.load_state_dict(torch.load(best_model_path))
    #     self.model.to(self.device)
    #     return self.model
    
    def reset_and_load_best_model(self, params, best_model_path):
    # Reinitialize the original model
    
        best_model = model.NetworkGraph(num_classes=params["num_classes"],
                                        network_config=params['network_config'], 
                                        network_gap=params['network_gap'])
        filtered_dict = {key: item for key, item in params['fn_dict'].items() if key in params['net_list']}
        best_model.create_functions(fn_dict=filtered_dict, net_list=params['net_list'])

        input_random = torch.randn(params['input_shape'])
        with torch.no_grad():
            _ = best_model(input_random)
        # Load the state dictionary of the best model into the new model
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.to(params['device'])

        return best_model

    def _initialize_scheduler(self, max_epochs):
        if self.params.get('lr_scheduler') == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            return ExponentialLR(self.optimizer, gamma=0.9)
        elif self.params.get('lr_scheduler') == 'reduce_on_plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            return ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)
        elif self.params.get('lr_scheduler') == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=0)
        elif self.params.get('lr_scheduler') == 'multistep':
            from torch.optim.lr_scheduler import MultiStepLR
            milestones = [0.5 * max_epochs, 0.75 * max_epochs]
            return MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        else:
            return None

    def train(self):
        max_epochs = self.params['max_epochs']
        epochs_to_eval = self.params.get('epochs_to_eval', max_epochs // 10)
        start_eval_epoch = max_epochs - epochs_to_eval
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []
        t0 = time.time()
        scheduler = self._initialize_scheduler(max_epochs)
        
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            training_losses.append(train_loss)
            training_accuracies.append(train_acc)
            self.logger.info(f"Epoch {epoch}/{max_epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            if epoch >= start_eval_epoch:
                val_loss, val_acc = self.evaluate(self.val_loader)
                validation_losses.append(val_loss)
                validation_accuracies.append(val_acc)
                self.logger.info(f"Epoch {epoch}/{max_epochs}: Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.best_epoch = epoch
                    self.no_improvement_epochs = 0
                    torch.save(self.model.state_dict(), self.best_model_path)
                    create_info_file(self.params['model_path'], {'best_accuracy': self.best_accuracy}, 'best_accuracy.txt')
                    # Save a checkpoint
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')
                    torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    self.no_improvement_epochs += 1

                if self.no_improvement_epochs >= self.early_stop_patience:
                    self.logger.info("Early stopping triggered.")
                    break

                self.update_scheduler(scheduler, metric=val_loss)
            else:
                self.update_scheduler(scheduler)
                
        total_training_time = time.time() - t0
        self.params['training_time'] = total_training_time
        
        # Load the best model after training
        self.model = self.reset_and_load_best_model(self.best_model_path)
        
        # Compute additional metrics
        model_metrics = metrics.ModelMetrics(self.model, device=self.params['device'])
        inference_images = next(iter(self.val_loader))[0][:10].to(self.params['device'])
        cuda_inference_time = model_metrics.measure_inference_time(inference_images)
        total_params = model_metrics.measure_parameters()
        total_flops = model_metrics.measure_flops(self.params['input_shape'])
        
        # Optionally compute confusion matrix and AUC on test set
        conf_matrix, auc, acc = self.compute_additional_metrics(self.test_loader)
        
        results = {
            'training_losses': training_losses,
            'training_accuracies': training_accuracies,
            'validation_losses': validation_losses,
            'validation_accuracies': validation_accuracies,
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'training_time': total_training_time,
            'cuda_inference_time': cuda_inference_time,
            'total_trainable_params': total_params,
            'total_flops': total_flops,
            'confusion_matrix': conf_matrix.tolist(),
            'auc': auc,
            'acc_test': acc
        }
        
        self.release_gpu_memory()
        return results
