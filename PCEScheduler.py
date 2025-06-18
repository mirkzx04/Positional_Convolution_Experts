import torch
import math

from torch.optim.lr_scheduler import _LRScheduler

class PCEScheduler(_LRScheduler):
    def __init__(self, optimizer, phase_epochs, base_lr, phase_multipliers, last_epoch=-1):
        """
        Custom scheduler for PCE training phases:
        - Phase 1 (EMA) : Higher LR for fast convergence
        - Phase 2 (Training) : Lower LR for fine-tuning

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to schedule.
            phase_epochs (list): List of epochs for each phase.
            base_lr (float): Base learning rate for the first phase.
            phase_multipliers (list): Multipliers for the learning rate in each phase.
            last_epoch (int): The index of the last epoch. Default is -1, which means the scheduler starts from the beginning.
        """

        self.phase_epochs = phase_epochs
        self.base_lr = base_lr
        self.phase_multipliers = phase_multipliers

        # Defines the boundaries for each phase based on the provided epochs
        self.phase_boundaries = [sum(phase_epochs[:i+1]) for i in range(len(phase_epochs))]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculate the learning rate for the current epoch based on the phase multiplier

        Process:
        1. Determine the current phase besed
        2. Calculate the phase-relative epoch
        3. Apply cosine ennealing inside the phase
        4. Return the final LR list
        """
        current_epoch = self.last_epoch

        # Determine current phase 
        phase = 0
        for i, boundary in enumerate(self.phase_boundaries):
            if current_epoch < boundary:
                phase = i
                break
            else:
                phase = len(self.phase_boundaries) - 1

        # Calculate phase-relative epoch
        phase_start = 0 if phase == 0 else self.phase_boundaries[phase - 1]
        phase_relative_epoch = current_epoch - phase_start 
        phase_length = self.phase_epochs[phase]

        # Base lr for this phase
        phase_lr = self.base_lr * self.phase_multipliers[phase]

        if phase_length > 0:
            # Normalize the progress and apply cosine annealing
            progress = phase_relative_epoch / phase_length
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

            final_lr = phase_lr * cosine_factor
        else:
            final_lr = phase_lr

        return [final_lr for _ in self.optimizer.param_groups]