"""
Training Coordinator for FHDP Pipeline Training

Coordinates the training process across multiple vehicles in a pipeline,
including model broadcasting, update aggregation, and training completion.
"""
import threading
import uuid
import torch
from typing import Dict, List, Optional, Any, Callable

from fhdp.core.types import Pipeline
from fhdp.core.cross_platform_comm import CrossPlatformMessage, PlatformBridge
from fhdp.utils.model_serialization import serialize_state_dict


class TrainingCoordinator:
    """
    Manages the training coordination process for pipeline training.

    Handles:
    - Broadcasting global models to vehicles
    - Collecting and aggregating model updates
    - Managing training rounds
    - Notifying training completion
    """

    def __init__(
        self,
        platform_bridge: PlatformBridge,
        global_model: torch.nn.Module,
        pipeline: Optional[Pipeline] = None
    ):
        """
        Initialize training coordinator.

        Args:
            platform_bridge: Cross-platform communication bridge
            global_model: Global model to be distributed
            pipeline: Active pipeline containing participating vehicles
        """
        self.platform_bridge = platform_bridge
        self.global_model = global_model
        self.pipeline = pipeline

        # Training state
        self.current_round = 0
        self.pending_updates: Dict[int, Dict[str, Dict]] = {}  # round -> vehicle_id -> update
        self.training_complete = False

        # Statistics
        self.stats = {
            'aggregations_performed': 0,
            'total_updates_received': 0,
            'training_rounds': 0
        }

        # Thread safety
        self.lock = threading.Lock()

        # Callbacks
        self.on_training_complete: Optional[Callable] = None
        self.on_round_complete: Optional[Callable[[int, Dict], None]] = None

    def set_pipeline(self, pipeline: Pipeline):
        """Set the active pipeline for training."""
        self.pipeline = pipeline

    def start_training_round(self, round_num: int, training_config: Dict[str, Any]):
        """
        Start a new training round.

        Args:
            round_num: Training round number
            training_config: Training configuration (epochs, batch_size, lr)
        """
        if not self.pipeline:
            raise ValueError("No pipeline set for training")

        self.current_round = round_num

        # Broadcast global model to all vehicles
        self.broadcast_global_model(round_num, training_config)

    def broadcast_global_model(self, round_num: int, training_config: Dict[str, Any]):
        """
        Broadcast the global model to all vehicles in the pipeline.

        Args:
            round_num: Training round number
            training_config: Training configuration to send to vehicles
        """
        # Serialize model state
        model_state = serialize_state_dict(self.global_model.state_dict())

        # Create broadcast message
        broadcast_msg = CrossPlatformMessage(
            message_id=str(uuid.uuid4()),
            source_id="server",
            target_id="",  # Broadcast to all
            message_type="global_model",
            payload={
                'round': round_num,
                'model_state': model_state,
                'training_config': training_config
            },
            requires_ack=False
        )

        # Send to all vehicles in pipeline
        for vehicle_id in self.pipeline.vehicles:
            broadcast_msg.target_id = vehicle_id
            target_endpoint = self.platform_bridge.message_router.routing_table.get(vehicle_id)
            if target_endpoint:
                self.platform_bridge.send_cross_platform_message(broadcast_msg)

    def receive_model_update(self, vehicle_id: str, update_data: Dict):
        """
        Receive a model update from a vehicle.

        Args:
            vehicle_id: ID of the vehicle sending the update
            update_data: Update payload containing model state and metrics
        """
        with self.lock:
            self.stats['total_updates_received'] += 1

        round_num = update_data.get('round', 0)

        # Store update
        if round_num not in self.pending_updates:
            self.pending_updates[round_num] = {}
        self.pending_updates[round_num][vehicle_id] = update_data

        # Check if we have updates from all vehicles for this round
        if self.pipeline:
            pipeline_vehicles = set(self.pipeline.vehicles)
            received_updates = set(self.pending_updates[round_num].keys())

            if received_updates == pipeline_vehicles:
                # Aggregate updates
                updates = [
                    self.pending_updates[round_num][vid]
                    for vid in self.pipeline.vehicles
                ]
                self.aggregate_model_updates(round_num, updates)

                # Clear pending updates for this round
                del self.pending_updates[round_num]

    def aggregate_model_updates(self, round_num: int, updates: List[Dict]):
        """
        Aggregate model updates from all vehicles.

        Args:
            round_num: Training round number
            updates: List of model updates from vehicles
        """
        # Simple averaging aggregation
        aggregated_state = {}
        num_updates = len(updates)

        for key in updates[0]['model_state'].keys():
            # Average parameters (deserialize list→Tensor if needed)
            tensors = [
                torch.tensor(update['model_state'][key])
                if isinstance(update['model_state'][key], list)
                else update['model_state'][key]
                for update in updates
            ]
            aggregated_state[key] = torch.mean(
                torch.stack(tensors),
                dim=0
            )

        # Update global model
        self.global_model.load_state_dict(aggregated_state)

        with self.lock:
            self.stats['aggregations_performed'] += 1
            self.stats['training_rounds'] += 1

        # Evaluate global model
        accuracy = self.evaluate_global_model(round_num)

        # Notify round completion
        if self.on_round_complete:
            self.on_round_complete(round_num, {'accuracy': accuracy})

    def evaluate_global_model(self, round_num: int) -> float:
        """
        Evaluate the global model (placeholder - actual implementation depends on task).

        Args:
            round_num: Training round number

        Returns:
            Mock accuracy (should be replaced with actual evaluation)
        """
        # Placeholder: mock accuracy that improves with rounds
        # In real implementation, this would evaluate on a validation set
        mock_accuracy = 0.7 + (round_num * 0.05)
        print(f"  Global model accuracy (round {round_num}): {mock_accuracy:.2%}")
        return mock_accuracy

    def complete_training(self):
        """
        Mark training as complete and notify all vehicles.
        """
        self.training_complete = True

        # Notify all vehicles
        self.notify_training_complete()

        # Call completion callback
        if self.on_training_complete:
            self.on_training_complete(self.stats)

    def notify_training_complete(self):
        """Send training complete notification to all vehicles."""
        if not self.pipeline:
            return

        for vehicle_id in self.pipeline.vehicles:
            msg = CrossPlatformMessage(
                message_id=str(uuid.uuid4()),
                source_id="server",
                target_id=vehicle_id,
                message_type="training_complete",
                payload={
                    'pipeline_id': self.pipeline.pipeline_id,
                    'total_rounds': self.current_round,
                    'final_stats': self.stats.copy()
                },
                requires_ack=False
            )

            try:
                self.platform_bridge.send_cross_platform_message(msg)
                print(f"✓ Sent training complete notification to {vehicle_id}")
            except Exception as e:
                print(f"✗ Failed to send training complete notification to {vehicle_id}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'current_round': self.current_round,
            'training_complete': self.training_complete,
            'stats': self.stats.copy(),
            'pending_updates': {
                round_num: list(updates.keys())
                for round_num, updates in self.pending_updates.items()
            }
        }
