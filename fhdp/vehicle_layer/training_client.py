"""
Training Client for FHDP Pipeline Training

Handles local training, model updates, and communication with the server
for vehicles participating in pipeline training.
"""
import threading
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

from fhdp.core.cross_platform_comm import CrossPlatformMessage, PlatformBridge
from fhdp.utils.model_serialization import serialize_state_dict, deserialize_state_dict


class TrainingClient:
    """
    Client-side training coordinator for pipeline training.

    Handles:
    - Receiving global model broadcasts
    - Local model training
    - Sending model updates to server
    - Training completion handling
    """

    def __init__(
        self,
        vehicle_id: str,
        platform_bridge: PlatformBridge,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        data_loader_fn: Optional[callable] = None
    ):
        """
        Initialize training client.

        Args:
            vehicle_id: Unique identifier for this vehicle
            platform_bridge: Cross-platform communication bridge
            model: PyTorch model to train
            optimizer: Optimizer for training
            criterion: Loss function
            data_loader_fn: Function to create data loader (num_samples, batch_size) -> DataLoader
        """
        self.vehicle_id = vehicle_id
        self.platform_bridge = platform_bridge
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_loader_fn = data_loader_fn

        # Training state
        self.current_pipeline_id: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.training_active = False
        self.current_round = 0

        # Statistics
        self.stats = {
            'training_sessions': 0,
            'epochs_completed': 0,
            'batches_processed': 0,
            'updates_sent': 0
        }

        # Thread safety
        self.lock = threading.Lock()

        # Training synchronization to prevent concurrent training
        self.training_lock = threading.Lock()
        self.latest_round_handled = 0

        # Callbacks
        self.on_training_complete: Optional[callable] = None

    def handle_global_model(self, message: CrossPlatformMessage):
        """
        Handle global model broadcast from server.

        Args:
            message: CrossPlatformMessage containing global model state
        """
        model_data = message.payload
        round_num = model_data['round']
        training_config = model_data['training_config']

        # Deduplicate: ignore if we've already handled this round
        if round_num <= self.latest_round_handled:
            return

        # Use lock to prevent concurrent training
        with self.training_lock:
            # Double-check after acquiring lock
            if round_num <= self.latest_round_handled:
                return

            # Mark this round as handled
            self.latest_round_handled = round_num

            print(f"\nReceived global model for round {round_num}")

            # Update local model
            self.model.load_state_dict(deserialize_state_dict(model_data['model_state']))

            # Start training
            self.train_locally(round_num, training_config)

    def train_locally(self, round_num: int, config: Dict[str, Any]):
        """
        Train model locally for one round.

        Args:
            round_num: Training round number
            config: Training configuration (epochs, batch_size, learning_rate)
        """
        print(f"→ Starting local training for round {round_num}...")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")

        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config['learning_rate']

        # Create data loader
        if self.data_loader_fn:
            train_loader = self.data_loader_fn(
                num_samples=200,
                batch_size=config['batch_size']
            )
        else:
            # Fallback: create simple mock data loader
            train_loader = self._create_mock_data_loader(
                num_samples=200,
                batch_size=config['batch_size']
            )

        # Training loop
        self.model.train()
        epoch_losses = []

        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                with self.lock:
                    self.stats['batches_processed'] += 1

                # Progress indicator
                if batch_idx % 5 == 0:
                    print(f"  Epoch {epoch+1}/{config['epochs']}, "
                          f"Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            epoch_losses.append(avg_epoch_loss)

            print(f"  Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss:.4f}")

            with self.lock:
                self.stats['epochs_completed'] += 1

        with self.lock:
            self.stats['training_sessions'] += 1

        print(f"✓ Local training completed for round {round_num}")
        print(f"  Average loss: {np.mean(epoch_losses):.4f}")

        # Send model update to server
        self.send_model_update(round_num, epoch_losses)

    def send_model_update(self, round_num: int, epoch_losses: List[float]):
        """
        Send model update to server.

        Args:
            round_num: Training round number
            epoch_losses: List of losses for each epoch
        """
        print(f"→ Sending model update for round {round_num}...")

        # Serialize model state
        model_state = serialize_state_dict(self.model.state_dict())

        update_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id=self.vehicle_id,
            target_id="server",
            message_type="model_update",
            payload={
                'round': round_num,
                'model_state': model_state,
                'loss': float(np.mean(epoch_losses)),
                'timestamp': time.time()
            },
            requires_ack=False
        )

        success = self.platform_bridge.send_cross_platform_message(update_msg)

        if success:
            with self.lock:
                self.stats['updates_sent'] += 1
            print(f"✓ Model update sent for round {round_num}")
        else:
            print(f"✗ Failed to send model update for round {round_num}")

    def handle_training_complete(self, message: CrossPlatformMessage):
        """
        Handle training complete notification from server.

        Args:
            message: CrossPlatformMessage containing training completion info
        """
        payload = message.payload
        print("\n" + "=" * 60)
        print("Training Complete Notification")
        print("=" * 60)
        print(f"Pipeline ID: {payload.get('pipeline_id')}")
        print(f"Total rounds completed: {payload.get('total_rounds')}")
        print(f"Final statistics: {payload.get('final_stats')}")

        # Call completion callback
        if self.on_training_complete:
            self.on_training_complete(payload)

    def set_pipeline_assignment(self, pipeline_id: str, stage: str):
        """
        Set the pipeline assignment for this vehicle.

        Args:
            pipeline_id: ID of the pipeline
            stage: Stage assigned to this vehicle
        """
        self.current_pipeline_id = pipeline_id
        self.current_stage = stage

    def stop_training(self):
        """Stop all training activities."""
        self.training_active = False

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'vehicle_id': self.vehicle_id,
            'current_pipeline_id': self.current_pipeline_id,
            'current_stage': self.current_stage,
            'latest_round_handled': self.latest_round_handled,
            'stats': self.stats.copy(),
            'training_active': self.training_active
        }

    def _create_mock_data_loader(self, num_samples: int = 100, batch_size: int = 32):
        """
        Create a mock data loader for testing.

        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for data loader

        Returns:
            Mock data loader
        """
        class MockDataLoader:
            def __init__(self, num_samples, batch_size):
                self.num_samples = num_samples
                self.batch_size = batch_size

            def __iter__(self):
                for i in range(0, self.num_samples, self.batch_size):
                    batch_size = min(self.batch_size, self.num_samples - i)

                    # Mock images (MNIST-like: 1x28x28)
                    images = torch.randn(batch_size, 1, 28, 28)
                    labels = torch.randint(0, 10, (batch_size,))

                    yield images, labels

            def __len__(self):
                return (self.num_samples + self.batch_size - 1) // self.batch_size

        return MockDataLoader(num_samples, batch_size)
