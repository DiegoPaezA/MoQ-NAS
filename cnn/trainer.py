""" Copyright (c) 2025, Diego Páez
* Licensed under the MIT license

- trainer module handles the training and evaluation of CNN models for evolutionary
    algorithms.
- It includes a base trainer class and a specialized ResNet trainer class.

"""

import os
import time

import torch
from medmnist import Evaluator
from sklearn.metrics import confusion_matrix
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                    MultiStepLR, ReduceLROnPlateau)

from cnn import fitness_utils, metrics, model, model_resnet
from util import create_info_file, init_log

TRAIN_TIMEOUT = 5400

current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

class BaseTrainer:
    """
    BaseTrainer is a base class for training, evaluating, and managing deep 
    learning models using PyTorch.
    It provides a unified interface for model training, validation, testing, 
    metric computation, and logging, with support for mixed precision training, 
    learning rate scheduling, and multi-objective fitness evaluation.
    Args:
        model_instance (torch.nn.Module): The neural network model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        params (dict): Dictionary of training parameters and configuration.
    Attributes:
        model (torch.nn.Module): The model being trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        params (dict): Training parameters.
        device (torch.device): Device for computation (CPU or CUDA).
        scaler (torch.cuda.amp.GradScaler): Mixed precision scaler.
        best_accuracy (float): Best validation accuracy achieved.
        best_validation_loss (float): Best validation loss achieved.
        best_epoch (int): Epoch with the best validation accuracy.
        best_model_path (str): Path to save the best model checkpoint.
        logger (logging.Logger): Logger for training events.
    Methods:
        _forward_pass(inputs, labels):
            Performs a forward pass with optional mixed precision and computes the loss.
        train_epoch():
            Trains the model for one epoch and returns average loss and accuracy.
        evaluate(loader):
            Evaluates the model on a given data loader and returns average loss and accuracy.
        compute_additional_metrics(loader):
            Computes additional metrics such as confusion matrix and AUC for multi-class tasks.
        update_scheduler(scheduler, metric=None):
            Updates the learning rate scheduler based on the specified policy.
        release_gpu_memory():
            Releases GPU memory by clearing the CUDA cache.
        reset_and_load_best_model(best_model_path):
            Reinitializes and loads the best model checkpoint.
        _initialize_scheduler(max_epochs):
            Initializes the learning rate scheduler based on configuration.
        should_evaluate(epoch, start_eval_epoch):
            Determines if evaluation should be performed at the current epoch.
        compute_mo_fitness(total_params, cuda_inference_time):
            Computes scalarized multi-objective fitness for model selection.
        get_final_metrics():
            Computes final model metrics such as inference time, parameter count, FLOPs, and memory usage.
        train(debug=False):
            Main training loop that manages training, validation, checkpointing, and metric computation.
    Notes:
        - Supports mixed precision training via torch.cuda.amp.
        - Handles both single-objective and multi-objective (e.g., accuracy, loss, parameters, inference time) optimization.
        - Designed for extensibility and integration with neural architecture search workflows.
    """
    def __init__(self, model_instance, criterion, optimizer, train_loader, val_loader, test_loader,
                params: dict):
        self.model = model_instance.to(params['device'])
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.params = params
        self.device = torch.device(params['device'])

        self.scaler = GradScaler(self.device.type, enabled=self.params.get('mixed_precision', False))
        self.no_improve_count = 0
        self.best_accuracy = 0.0
        self.best_validation_loss = float('inf')
        self.best_epoch = 0
        self.best_model_path = os.path.join(self.params['model_path'], 'best_model.pth')
        if not os.path.exists(self.params['model_path']):
            os.makedirs(self.params['model_path'])

        # Initialize the logger
        phase = self.params.get('phase', 'evolution')
        if phase == 'retrain':
            log_file = os.path.join(log_directory, "retrain.log")
        elif phase == 'evolution':
            log_file = os.path.join(log_directory, "evolution.log")
        else:
            log_file = os.path.join(log_directory, "default.log")

        self.logger = init_log(log_level="INFO", name=__name__, file_path=log_file)

    def _forward_pass(self, inputs, labels):
        """
        Performs a forward pass through the model with the given inputs and labels.
        Moves the input data and labels to the appropriate device, adjusts label shape and type for multi-class tasks,
        and computes the model outputs and loss. Supports mixed precision inference if enabled in parameters.
        Args:
            inputs (torch.Tensor): Input data batch.
            labels (torch.Tensor): Corresponding labels for the input data.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the model outputs, computed loss, and processed labels.
        """
        # Move data to device
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Adjust labels for multi-class tasks
        task = self.params.get('task', 'classification')
        if task == 'multi-class':
            labels = labels.squeeze().long()
        
        # Run forward pass with mixed precision if enabled
        with autocast(self.device.type, dtype=torch.float16, enabled=self.params.get('mixed_precision', False)):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
        
        return outputs, loss, labels
    
    def train_epoch(self):
        """
        Trains the model for one epoch using the training data loader.
        Performs a forward pass, computes the loss, applies mixed precision scaling for backpropagation,
        updates the optimizer, and tracks training loss and accuracy.
        Returns:
            tuple: A tuple containing:
                - avg_loss (float): The average loss over the epoch.
                - accuracy (float): The training accuracy (in percentage) over the epoch.
        """
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
        """
        Evaluates the model on the provided data loader.
        Args:
            loader (torch.utils.data.DataLoader): DataLoader providing the evaluation dataset.
        Returns:
            tuple: A tuple containing:
                - avg_loss (float): The average loss over the evaluation dataset.
                - accuracy (float): The classification accuracy (in percentage) over the evaluation dataset.
        Notes:
            - The model is set to evaluation mode during this process.
            - No gradients are computed.
        """
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
        Computes additional evaluation metrics for the model on a given data loader.
        This method evaluates the model in evaluation mode over the provided data loader,
        calculating the average loss, accuracy, confusion matrix, and, if applicable,
        AUC and evaluator accuracy for multi-class tasks.
        Args:
            loader (torch.utils.data.DataLoader): DataLoader providing the dataset to evaluate.
        Returns:
            tuple:
                - test_acc (float): The accuracy of the model on the dataset (in percentage).
                - avg_loss (float): The average loss over the dataset.
                - conf_matrix (np.ndarray): Confusion matrix of true vs. predicted labels.
                - auc (float): Area Under the Curve (AUC) score for multi-class tasks (0.0 otherwise).
                - evaluator_acc (float): Evaluator accuracy for multi-class tasks (0.0 otherwise).
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
        """
        Updates the learning rate scheduler based on the specified strategy.

        If the scheduler is set to 'reduce_on_plateau' and a metric is provided, 
        the scheduler's step method is called with the metric to adjust the learning rate 
        based on the metric's value (typically validation loss). Otherwise, the scheduler's 
        step method is called without arguments to perform a standard update.

        Args:
            scheduler (torch.optim.lr_scheduler._LRScheduler or None): The learning rate 
            scheduler to update.
            metric (float, optional): The value of the monitored metric (e.g., validation loss) 
                used when the scheduler is 'reduce_on_plateau'. Defaults to None.

        """
        if scheduler is not None:
            if self.params.get('lr_scheduler') == 'reduce_on_plateau' and metric is not None:
                scheduler.step(metric)
            else:
                scheduler.step()

    def release_gpu_memory(self):
        """
        Releases unused GPU memory by clearing the CUDA cache.

        This method checks if CUDA is available and, if so, clears the GPU cache 
        to free up unused memory.
        If CUDA is not available, the method exits without performing any action.
        """
        if not torch.cuda.is_available():
            #self.logger.info("CUDA is not available. No GPU memory to release.")
            return
        torch.cuda.empty_cache()
        #self.logger.info("GPU cache cleared.")

    def reset_and_load_best_model(self, best_model_path):
        """
        Reinitializes the model and loads the weights from the specified best model checkpoint.
        This method creates a new instance of the model using the current parameters,
        filters and assigns the appropriate functions, performs a dummy forward pass to
        initialize the model's layers, loads the saved state dictionary from the given
        checkpoint path, moves the model to the specified device, and returns the loaded model.
        Args:
            best_model_path (str): Path to the checkpoint file containing the best model's 
            state_dict.
        Returns:
            torch.nn.Module: The reinitialized model with weights loaded from the checkpoint.
        """
        # Reinitialize the model and load weights from the best model checkpoint.
        best_model = model.NetworkGraph(num_classes=self.params["num_classes"],
                                        input_shape=self.params['input_shape'],
                                        network_config=self.params['network_config'],
                                        backbone_name=self.params['backbone_name'],
                                        backbone_percentage=self.params['backbone_percentage'])
        
        filtered_dict = {key: item for key, item in self.params['fn_dict'].items() if key in self.params['net_list']}
        best_model.create_functions(fn_dict=filtered_dict, net_list=self.params['net_list'])
        input_random = torch.randn(self.params['input_shape'])
        with torch.no_grad():
            _ = best_model(input_random)
        best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
        best_model.to(self.params['device'])
        return best_model

    def _initialize_scheduler(self, max_epochs):
        """
        Initializes and returns a learning rate scheduler for the optimizer based on the specified parameters.

        Args:
            max_epochs (int): The maximum number of training epochs, used for certain schedulers.

        Returns:
            torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.ReduceLROnPlateau or None:
                The initialized learning rate scheduler object, or None if no scheduler is specified.

        Supported schedulers:
            - 'exponential': ExponentialLR with gamma=0.9.
            - 'reduce_on_plateau': ReduceLROnPlateau with patience=5 and factor=0.1.
            - 'cosine': CosineAnnealingLR with T_max=max_epochs and eta_min=0.
            - 'multistep': MultiStepLR with milestones at 50% and 75% of max_epochs, gamma=0.1.
        """
        if self.params.get('lr_scheduler') == 'exponential':
            return ExponentialLR(self.optimizer, gamma=0.9)
        elif self.params.get('lr_scheduler') == 'reduce_on_plateau':
            return ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)
        elif self.params.get('lr_scheduler') == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=0)
        elif self.params.get('lr_scheduler') == 'multistep':
            milestones = [int(0.5 * max_epochs), int(0.75 * max_epochs)]
            return MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        else:
            return None

    def should_evaluate(self, epoch, start_eval_epoch):
        """
        Determines whether evaluation should be performed at the given epoch.

        Args:
            epoch (int): The current epoch number.
            start_eval_epoch (int): The epoch number after which evaluation should start during the 'evolution' phase.

        Returns:
            bool: True if evaluation should be performed based on the current phase and epoch, False otherwise.

        Phases:
            - 'evolution': Evaluation starts after 'start_eval_epoch'.
            - 'retrain' or 'resnet': Evaluation is always performed.
        """
        phase = self.params.get('phase')
        if phase == 'evolution' and epoch > start_eval_epoch:
            return True
        elif phase == 'retrain' or phase == 'resnet':
            return True
        return False
    
    def compute_mo_fitness(self, total_params, cuda_inference_time):
        """
        Computes the multi-objective fitness values for a neural architecture search process.
        This method calculates two fitness metrics:
        1. A scalarized multi-objective fitness value using the specified base metric (accuracy or loss),
            total number of parameters, and CUDA inference time.
        2. A validation loss-based fitness value, scaled to a percentage.
        Args:
            total_params (int): The total number of parameters in the model.
            cuda_inference_time (float): The model's inference time on CUDA (in seconds or milliseconds).
        Returns:
            tuple:
                - scalar_multi_objective (float): The scalarized multi-objective fitness value.
                - fitness_val_loss (float): The fitness value based on validation loss, scaled to [0, 100].
        Raises:
            ValueError: If the specified fitness metric is invalid or not supported.
        """
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
        Compute and return final evaluation metrics for the current model.
        This method calculates the following metrics:
        - CUDA inference time (in microseconds) using a small batch from the validation loader.
        - Total number of trainable parameters in the model.
        - Total number of floating point operations (FLOPs) based on the input shape.
        - Model memory usage in megabytes (MB) for the given input shape.
            tuple: A tuple containing:
                - cuda_inference_time (float): Inference time on CUDA in microseconds.
                - total_params (int): Total number of trainable parameters.
                - total_flops (int): Total FLOPs for the given input shape.
                - model_memory_usage (float): Model memory usage in MB.
        
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
        patience_max = self.params.get('patience_retrain', max_epochs)
        base_fraction = self.params.get('delta_fraction', 0.005)
        start_eval_epoch = max_epochs - epochs_to_eval
        training_losses, training_accuracies = [], []
        validation_losses, validation_accuracies = [], []
        t0 = time.time()
        if self.params.get('phase') == 'retrain':
            self.logger.info("Retraining evolved model %s ...", self.params['experiment_path'])
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
                    create_info_file(self.params['model_path'], {'best_accuracy': self.best_accuracy}, 'best_accuracy.txt')
                    if self.params.get('phase') == 'retrain':
                        torch.save(self.model.state_dict(), self.best_model_path)
                        
                # Dynamically compute min_delta as a fraction of best_validation_loss
                if self.best_validation_loss == float('inf'):
                    self.best_validation_loss = val_loss
                    no_improve_count = 0
                    continue
                
                min_delta = self.best_validation_loss * base_fraction
                if (self.best_validation_loss - val_loss) > min_delta:
                    self.best_validation_loss = val_loss
                    self.best_epoch = epoch
                    no_improve_count = 0
                    # if self.params.get('phase') == 'retrain':
                    #     torch.save(self.model.state_dict(), self.best_model_path)
                        
                else:
                    # no sufficient improvement
                    no_improve_count += 1
                    # if epoch % 5 == 0:
                    #     self.logger.info(
                    #         "Epoch [%d/%d] - No val_loss drop > %.4f for %d/%d evals",
                    #         epoch, max_epochs, min_delta, no_improve_count, patience_max
                    #     )
                    if no_improve_count >= patience_max and self.params.get('phase') == 'retrain':
                        self.logger.info("Early stopping at epoch %d", epoch)
                        break
                
                if self.params.get('phase') == 'retrain':
                    self.update_scheduler(scheduler, metric=val_loss)
                    if epoch % 25 == 0:
                        self.logger.info("Experiment: %s: Epoch [%d/%d] - Train Loss: %.2f - Val Loss: %.2f - Val Acc: %.2f%%",
                            self.params['experiment_path'], epoch, max_epochs, train_loss, val_loss, val_acc
                        )
                if debug:
                    if epoch >= start_eval_epoch:
                        self.logger.info("Epoch [%d/%d] - Training Loss: %.4f - Validation Loss: %.4f - Validation Accuracy: %.2f%%",
                            epoch, max_epochs, train_loss, val_loss, val_acc
                        )
                    elif epoch % 5 == 0:
                        self.logger.info("Epoch [%d/%d] - Training Loss: %.4f - Training Accuracy: %.2f%%",
                            epoch, max_epochs, train_loss, train_acc)

        total_training_time = time.time() - t0
        self.params['training_time'] = total_training_time
        
        # Handle retrain phase: load best model and compute test metrics.
        phase = self.params.get('phase')
        if (phase == 'retrain' or phase == 'resnet') and self.test_loader is not None:
            self.model = self.reset_and_load_best_model(self.best_model_path)
            test_acc, test_loss, conf_matrix, auc, acc = self.compute_additional_metrics(self.test_loader)
            self.logger.info("Experiment: %s - Test loss: %.2f - Test accuracy: %.2f%%",
                self.params['experiment_path'], test_loss, test_acc)
        else:
            test_acc, test_loss, conf_matrix, auc, acc = None, None, None, None, None
        
        cuda_inference_time, total_params, total_flops, model_memory_usage = self.get_final_metrics()

        # If in evolution phase, compute the fitness metrics.
        if self.params.get('phase') == 'evolution':
            scalar_multi_objective, fitness_val_loss = self.compute_mo_fitness(total_params, cuda_inference_time)
        else:
            scalar_multi_objective, fitness_val_loss = None, None
        
        self.params['total_params'] = total_params
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
            'total_params': total_params,
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
    """
    ResNetTrainer is a specialized trainer class for training and evaluating ResNet models.
    Args:
        model_instance (torch.nn.Module): The neural network model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        params (dict): Dictionary of training parameters and model configuration.
        logger (optional): Logger object for tracking training progress.
    Attributes:
        model_flag (str): Indicates which ResNet variant to use (e.g., 'resnet18', 'resnet50').
    Methods:
        reset_and_load_best_model(best_model_path):
            Instantiates the specified ResNet model architecture and loads the state dictionary
            from the provided checkpoint path.
            Args:
                best_model_path (str): Path to the saved model checkpoint.
            Returns:
                torch.nn.Module: The ResNet model loaded with the best checkpoint weights.
            Raises:
                ValueError: If an unsupported model_flag is provided.
    """

    def __init__(self, model_instance, criterion, optimizer, train_loader, val_loader, test_loader,
                params: dict, logger=None):
        # Call the parent constructor
        super().__init__(model_instance, criterion, optimizer, train_loader, val_loader, test_loader, params, logger)
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
        best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
        best_model.to(self.params['device'])
        return best_model