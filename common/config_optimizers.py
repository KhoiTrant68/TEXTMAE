from torch import optim


def configure_optimizers(model, args):
    """Return two optimizers"""
    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters

    assert len(inter_params) == 0

    optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.lr,
    )
    aux_optimizer = optim.AdamW(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_lr,
    )
    return optimizer, aux_optimizer
