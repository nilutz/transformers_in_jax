import os
from pathlib import Path

import orbax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from torch.utils.tensorboard import SummaryWriter

CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "checkpoints"
LOG_PATH = Path(__file__).parent.parent.parent / "runs"


if not CHECKPOINT_PATH.is_dir():
    Path(CHECKPOINT_PATH).mkdir()

if not LOG_PATH.is_dir():
    Path(LOG_PATH).mkdir()


class DefaultTrainer:
    def __init__(
        self,
        model,
        model_name: str,
        config,
        init_batch: tuple,
        seed=42,
    ):
        """
        Args:
            model:  A Transformers model
            model_name: A name for the model for storage
            init_batch: tuple containing src, tgt batch
        """
        self.model = model
        self.model_name = model_name
        self.config = config
        self.seed: int = seed

        self.log_dir: str = os.path.join(LOG_PATH, self.model_name)
        self.logger: SummaryWriter = SummaryWriter(log_dir=self.log_dir)
        self.checkpoint_dir: str = os.path.join(CHECKPOINT_PATH, self.model_name)

        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.ckptr = ocp.CheckpointManager(
            self.checkpoint_dir,
            self.orbax_checkpointer,
            options=options,
        )

        self.create_functions()
        self.init_model(init_batch)

    def get_loss_function(self):
        raise NotImplementedError

    def create_functions(self):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def save_model(self, step: int = 0):
        print("save1")
        ref_ckpt = {"state": self.state}
        print("save2", type(ref_ckpt))
        save_args = orbax_utils.save_args_from_target(ref_ckpt)
        print("save3", type(save_args))
        self.ckptr.save(step, ref_ckpt, save_kwargs={"save_args": save_args})
        print("save4", type(save_args))
        self.ckptr.wait_until_finished()
        print("save5", type(save_args))

    def load_model(self, step: int = None):
        if step is None:
            step = self.ckptr.latest_step()

        ref_ckpt = {"state": self.state}
        restore_args = orbax_utils.restore_args_from_target({"restore_args": ref_ckpt})
        params = self.ckptr.restore(step, restore_kwargs=restore_args, items=ref_ckpt)
        self.ckptr.wait_until_finished()

        self.state = params["state"]

    def checkpoint_exists(self, epoch: int = 0) -> bool:
        # Check whether a pretrained model exist for this Transformer
        return os.path.isfile(
            os.path.join(CHECKPOINT_PATH, f"{self.model_name}", str(epoch), "default")
        )

    def eval_model(self):
        raise NotImplementedError
