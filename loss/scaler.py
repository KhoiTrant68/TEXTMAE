import torch
from torch import inf


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        # Initialize GradScaler from torch.cuda.amp
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        # Scale the loss and perform backward pass
        self._scaler.scale(loss).backward(create_graph=create_graph)

        if update_grad:
            # Unscaled gradients and optionally clip them
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # Unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                # Calculate the norm of the gradients
                norm = get_grad_norm_(parameters)

            # Step optimizer and update scaler
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None

        return norm

    def state_dict(self):
        # Get the state dictionary of the scaler
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        # Load the state dictionary into the scaler
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    # Compute the gradient norm
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device

    if norm_type == inf:
        # Return the max norm if norm_type is infinity
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # Return the norm based on the specified norm_type
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )

    return total_norm
