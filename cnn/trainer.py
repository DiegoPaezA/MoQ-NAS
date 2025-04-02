""" Copyright (c) 2025, Diego Páez
* Licensed under the MIT license

- trainer module handles the training and evaluation of CNN models for evolutionary
    algorithms.
- It includes a base trainer class and a specialized ResNet trainer class.

"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from util import create_info_file, init_log, load_yaml
from cnn import metrics, model, fitness_utils, model_resnet
from sklearn.metrics import confusion_matrix
from medmnist import Evaluator

TRAIN_TIMEOUT = 5400

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
        self.best_validation_loss = float('inf')
        self.best_epoch = 0
        self.best_model_path = os.path.join(self.params['model_path'], 'best_model.pth')
        if not os.path.exists(self.params['model_path']):
            os.makedirs(self.params['model_path'])

    def _forward_pass(self, inputs, labels):
        """
        Common forward pass: moves inputs/labels to device, adjusts labels for multi-class,
        runs the model under autocast, and computes the loss.
        """
        # Move data to device
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Adjust labels for multi-class tasks
        task = self.params.get('task', 'classification')
        if task == 'multi-class':
            labels = labels.squeeze().long()
        
        # Run forward pass with mixed precision if enabled
        with autocast(dtype=torch.float16, enabled=self.params.get('mixed_precision', False)):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
        
        return outputs, loss, labels
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs, loss, labels = self._forward_pass(inputs, labels)
            
            # Backpropagation with mixed precision scaling
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
                outputs, loss, labels = self._forward_pass(inputs, labels)
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        avg_loss = epoch_loss / len(loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def compute_additional_metrics(self, loader):
        """
        Compute additional metrics such as loss, accuracy, confusion matrix, and,
        for multi-class tasks, AUC. Uses a shared forward pass helper.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        y_scores = []  # For storing softmax outputs if needed
        
        task = self.params.get('task', 'classification')
        
        with torch.no_grad():
            for inputs, labels in loader:
                # Use the helper to move data, perform forward pass, and compute loss
                outputs, loss, labels = self._forward_pass(inputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Collect labels and predictions for the confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # For multi-class tasks, store softmax outputs
                if task == 'multi-class':
                    y_scores.append(outputs.softmax(dim=-1).cpu())
        
        avg_loss = total_loss / len(loader)
        test_acc = 100 * correct / total
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        auc = 0.0
        evaluator_acc = 0.0
        if task == 'multi-class' and y_scores:
            # Concatenate softmax outputs and evaluate AUC
            y_score = torch.cat(y_scores, dim=0).numpy()
            evaluator = Evaluator(self.params['dataset'], split='test', root=self.params['data_path'])
            auc, evaluator_acc = evaluator.evaluate(y_score)
        
        return test_acc, avg_loss, conf_matrix, auc, evaluator_acc

    def update_scheduler(self, scheduler, metric=None):
        if scheduler is not None:
            if self.params.get('lr_scheduler') == 'reduce_on_plateau' and metric is not None:
                scheduler.step(metric)
            else:
                scheduler.step()

    def release_gpu_memory(self):
        if not torch.cuda.is_available():
            #self.logger.info("CUDA is not available. No GPU memory to release.")
            return
        torch.cuda.empty_cache()
        #self.logger.info("GPU cache cleared.")

    def reset_and_load_best_model(self, best_model_path):
        # Reinitialize the model and load weights from the best model checkpoint.
        best_model = model.NetworkGraph(num_classes=self.params["num_classes"],
                                        network_config=self.params['network_config'], 
                                        network_gap=self.params['network_gap'])
        filtered_dict = {key: item for key, item in self.params['fn_dict'].items() if key in self.params['net_list']}
        best_model.create_functions(fn_dict=filtered_dict, net_list=self.params['net_list'])
        input_random = torch.randn(self.params['input_shape'])
        with torch.no_grad():
            _ = best_model(input_random)
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.to(self.params['device'])
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
            milestones = [int(0.5 * max_epochs), int(0.75 * max_epochs)]
            return MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        else:
            return None

    def should_evaluate(self, epoch, start_eval_epoch):
        phase = self.params.get('phase')
        if phase == 'evolution' and epoch > start_eval_epoch:
            return True
        elif phase == 'retrain' or phase == 'resnet':
            return True
        return False
    
    def compute_mo_fitness(self, total_params, cuda_inference_time):
        fitness_metric = self.params.get('fitness_metric')
        mo_base_metric = self.params.get('mo_metric_base')
        if fitness_metric == 'best_accuracy' or (fitness_metric == 'scalar_multi_objective' and mo_base_metric == 'accuracy'):
            metric_value = self.best_accuracy
            metric_type = 'accuracy'
        elif fitness_metric == 'best_loss' or (fitness_metric == 'scalar_multi_objective' and mo_base_metric == 'loss'):
            metric_value = self.best_validation_loss
            metric_type = 'loss'
        else:
            raise ValueError(f"Invalid fitness_metric: {fitness_metric}")
        
        # Scalarized multi-objective function.
        scalar_multi_objective = fitness_utils.mofitness(metric_value=metric_value,
                                                        params=total_params,
                                                        inference_time=cuda_inference_time,
                                                        T_p=self.params['max_params'],
                                                        T_t=self.params['max_inference_time'],
                                                        metric_type=metric_type)
        fitness_val_loss = (1 / (1 + self.best_validation_loss)) * 100.0
        return scalar_multi_objective, fitness_val_loss
    
    def get_final_metrics(self):
        """
        Compute final metrics for the current model:
        - CUDA inference time (in microseconds)
        - Total trainable parameters
        - Total FLOPs (as measured on the input shape)
        - Model memory usage (in MB)
        
        Returns:
            tuple: (cuda_inference_time, total_params, total_flops, model_memory_usage)
        """
        model_metrics = metrics.ModelMetrics(self.model, device=self.params['device'])
        # Get a small batch from the validation loader for measuring inference time.
        inference_images = next(iter(self.val_loader))[0][:10].to(self.params['device'])
        cuda_inference_time = model_metrics.measure_inference_time(inference_images)
        total_params = model_metrics.measure_parameters()
        total_flops = model_metrics.measure_flops(self.params['input_shape'])
        model_memory_usage = model_metrics.measure_memory(self.params['input_shape']) / (1024 ** 2)  # Convert bytes to MB
        return cuda_inference_time, total_params, total_flops, model_memory_usage

    def train(self, debug=False):
        max_epochs = self.params['max_epochs']
        epochs_to_eval = self.params['epochs_to_eval']
        start_eval_epoch = max_epochs - epochs_to_eval
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []
        t0 = time.time()
        if self.params.get('phase') == 'retrain':
            scheduler = self._initialize_scheduler(max_epochs)
                    
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            training_losses.append(train_loss)
            training_accuracies.append(train_acc)
            
            if epoch < start_eval_epoch and (time.time() - t0) > TRAIN_TIMEOUT and self.params.get('phase') != 'retrain':
                self.logger.info("Timeout reached")
                raise TimeoutError()
                        
            if self.should_evaluate(epoch, start_eval_epoch):
                val_loss, val_acc = self.evaluate(self.val_loader)
                validation_losses.append(val_loss)
                validation_accuracies.append(val_acc)                
                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), self.best_model_path)
                    create_info_file(self.params['model_path'], {'best_accuracy': self.best_accuracy}, 'best_accuracy.txt')
                    
                if val_loss < self.best_validation_loss:
                    self.best_validation_loss = val_loss
                
                if self.params.get('phase') == 'retrain':
                    self.update_scheduler(scheduler, metric=val_loss)
                    if epoch % 25 == 0:
                        self.logger.info(f"Epoch [{epoch}/{max_epochs}] - Train Loss: {train_loss:.2f} - Val Loss: {val_loss:.2f} - Val Acc: {val_acc:.2f}%")
            if debug:
                if epoch >= start_eval_epoch:
                    print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.2f}%")
                elif epoch % 5 == 0:
                    print(f"Epoch [{epoch}/{max_epochs}] - Training Loss: {train_loss:.4f} - Training Accuracy: {train_acc:.2f}%")

        total_training_time = time.time() - t0
        self.params['training_time'] = total_training_time
        
        # Handle retrain phase: load best model and compute test metrics.
        phase = self.params.get('phase')
        if (phase == 'retrain' or phase == 'resnet') and self.test_loader is not None:
            self.model = self.reset_and_load_best_model(self.best_model_path)
            test_acc, test_loss, conf_matrix, auc, acc = self.compute_additional_metrics(self.test_loader)
        else:
            test_acc, test_loss, conf_matrix, auc, acc = None, None, None, None, None
        
        cuda_inference_time, total_params, total_flops, model_memory_usage = self.get_final_metrics()

        # If in evolution phase, compute the fitness metrics.
        if self.params.get('phase') == 'evolution':
            scalar_multi_objective, fitness_val_loss = self.compute_mo_fitness(total_params, cuda_inference_time)
        else:
            scalar_multi_objective, fitness_val_loss = None, None
        
        self.params['total_trainable_params'] = total_params
        self.params['cuda_inference_time'] = cuda_inference_time
        self.params['model_memory_usage'] = model_memory_usage
        self.params['total_flops'] = total_flops
        self.params['best_accuracy'] = self.best_accuracy
        self.params['best_validation_loss'] = self.best_validation_loss
        self.params['fitness_val_loss'] = fitness_val_loss
        self.params['scalar_multi_objective'] = scalar_multi_objective
        self.params['test_accuracy'] = test_acc
        self.params['test_loss'] = test_loss

        
        create_info_file(self.params['model_path'], self.params, 'training_params.txt')
        
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
            'model_memory_usage': model_memory_usage,
            'fitness_val_loss': fitness_val_loss,
            'scalar_multi_objective': scalar_multi_objective,
            'total_flops': total_flops,
            'confusion_matrix': conf_matrix.tolist() if conf_matrix is not None else None,
            'auc_score': auc,
            'acc_medmnist': acc,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            
        }
        
        self.release_gpu_memory()
        return results
    
class ResNetTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, test_loader,
                params: dict, logger=None):
        # Call the parent constructor
        super().__init__(model, criterion, optimizer, train_loader, val_loader, test_loader, params, logger)
        # Use a parameter to decide which ResNet to use (default to 'resnet18')
        self.model_flag = self.params.get('model_flag', 'resnet18')

    def reset_and_load_best_model(self, best_model_path):
        """
        For ResNet models, we instantiate the model using our model_resnet module and
        load the state dictionary from the best checkpoint.
        """
        # Define which ResNet classes are available.
        model_classes = {
            'resnet18': model_resnet.ResNet18,
            'resnet50': model_resnet.ResNet50
        }
        if self.model_flag not in model_classes:
            raise ValueError(f"Unsupported model_flag: {self.model_flag}")
        
        # Instantiate the chosen ResNet model.
        best_model = model_classes[self.model_flag](
            in_channels=self.params['input_shape'][1],
            num_classes=self.params['num_classes']
        )
        # Load the saved state.
        best_model.load_state_dict(torch.load(best_model_path))
        best_model.to(self.params['device'])
        return best_model