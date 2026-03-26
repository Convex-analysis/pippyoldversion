"""
Pipeline Training Manager for FHDP System

High-level manager that orchestrates the entire pipeline training process,
including coordination with TrainingCoordinator and communication with vehicles.
"""
import threading
import time
from typing import Dict, Optional, Any, Callable

from fhdp.core.types import Pipeline
from fhdp.core.cross_platform_comm import PlatformBridge
from fhdp.edge_server.training_coordinator import TrainingCoordinator


class PipelineTrainingManager:
    """
    Manages the complete pipeline training lifecycle.

    This class provides a high-level interface for:
    - Starting and managing pipeline training
    - Coordinating training rounds
    - Handling training completion
    - Providing status and statistics
    """

    def __init__(
        self,
        platform_bridge: PlatformBridge,
        global_model,
        num_rounds: int = 3
    ):
        """
        Initialize pipeline training manager.

        Args:
            platform_bridge: Cross-platform communication bridge
            global_model: Global model to train
            num_rounds: Number of training rounds to perform
        """
        self.platform_bridge = platform_bridge
        self.global_model = global_model
        self.num_rounds = num_rounds

        # Training coordinator
        self.training_coordinator = TrainingCoordinator(
            platform_bridge=platform_bridge,
            global_model=global_model
        )

        # Set up callbacks
        self.training_coordinator.on_training_complete = self._on_training_complete
        self.training_coordinator.on_round_complete = self._on_round_complete

        # State
        self.active_pipeline: Optional[Pipeline] = None
        self.training_started = False
        self.training_complete = False

        # Thread safety
        self.lock = threading.Lock()

    def set_pipeline(self, pipeline: Pipeline):
        """
        Set the active pipeline for training.

        Args:
            pipeline: Pipeline configuration with participating vehicles
        """
        self.active_pipeline = pipeline
        self.training_coordinator.set_pipeline(pipeline)

    def start_training(self, training_config: Optional[Dict[str, Any]] = None):
        """
        Start pipeline training.

        Args:
            training_config: Training configuration (epochs, batch_size, lr)
        """
        if not self.active_pipeline:
            raise ValueError("No pipeline set. Call set_pipeline() first.")

        if self.training_started:
            print("Training already started")
            return

        with self.lock:
            self.training_started = True

        print("\n" + "=" * 60)
        print("Pipeline Training Started")
        print("=" * 60)
        print(f"Pipeline ID: {self.active_pipeline.pipeline_id}")
        print(f"Vehicles: {self.active_pipeline.vehicles}")
        print(f"Stages: {self.active_pipeline.stages}")
        print(f"Training rounds: {self.num_rounds}")
        print("=" * 60 + "\n")

        # Default training configuration
        if training_config is None:
            training_config = {
                'epochs': 2,
                'batch_size': 32,
                'learning_rate': 0.001
            }

        # Start first round
        self._start_round(1, training_config)

    def _start_round(self, round_num: int, training_config: Dict[str, Any]):
        """
        Start a training round.

        Args:
            round_num: Round number
            training_config: Training configuration
        """
        print(f"\n{'='*20} Round {round_num} {'='*20}")

        # Start the round through coordinator
        self.training_coordinator.start_training_round(round_num, training_config)

    def _on_round_complete(self, round_num: int, metrics: Dict[str, Any]):
        """
        Callback when a round completes.

        Args:
            round_num: Round number
            metrics: Round metrics (e.g., accuracy)
        """
        print(f"✓ Round {round_num} completed")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")

        # Start next round if not done
        if round_num < self.num_rounds:
            time.sleep(2.0)
            # Create a new thread for the next round
            threading.Thread(
                target=self._start_round,
                args=(round_num + 1, {'epochs': 2, 'batch_size': 32, 'learning_rate': 0.001}),
                daemon=True
            ).start()
        else:
            # All rounds complete
            self.training_coordinator.complete_training()

    def _on_training_complete(self, final_stats: Dict[str, Any]):
        """
        Callback when training completes.

        Args:
            final_stats: Final training statistics
        """
        with self.lock:
            self.training_complete = True

        print("\n" + "=" * 60)
        print("Pipeline Training Completed")
        print("=" * 60)
        self._print_summary(final_stats)

    def receive_model_update(self, vehicle_id: str, update_data: Dict):
        """
        Forward model update to training coordinator.

        Args:
            vehicle_id: ID of the vehicle sending the update
            update_data: Update payload
        """
        self.training_coordinator.receive_model_update(vehicle_id, update_data)

    def _print_summary(self, stats: Dict[str, Any]):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Vehicles in pipeline: {len(self.active_pipeline.vehicles) if self.active_pipeline else 0}")
        print(f"Training rounds: {stats.get('training_rounds', 0)}")
        print(f"Aggregations performed: {stats.get('aggregations_performed', 0)}")
        print(f"Total updates received: {stats.get('total_updates_received', 0)}")
        print("=" * 60)

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        coordinator_status = self.training_coordinator.get_status()

        return {
            'pipeline_id': self.active_pipeline.pipeline_id if self.active_pipeline else None,
            'vehicles': self.active_pipeline.vehicles if self.active_pipeline else [],
            'num_rounds': self.num_rounds,
            'training_started': self.training_started,
            'training_complete': self.training_complete,
            'coordinator_status': coordinator_status
        }

    def reset(self):
        """Reset training state for a new training session."""
        with self.lock:
            self.training_started = False
            self.training_complete = False
            self.active_pipeline = None

        # Reset coordinator
        self.training_coordinator.current_round = 0
        self.training_coordinator.pending_updates = {}
        self.training_coordinator.training_complete = False
