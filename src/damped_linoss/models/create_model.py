import jax
import equinox as eqx

from damped_linoss.models.RNN import BasicRNN, StackedRNN
from damped_linoss.models.S5 import S5
from damped_linoss.models.LRU import LRU
from damped_linoss.models.LinOSS import LinOSS


def safe_load(data, key, dtype=None):
    val = data.get(key, None)
    if val is None:
        raise KeyError(f"Key {key} does not exist")
    if dtype is not None:
        val = dtype(val)
    return val


def create_model(
    hyperparameters: dict,
    key: jax.Array,
):
    model_name = safe_load(hyperparameters, "model_name", str)

    if model_name == "RNN":
        stacked = safe_load(hyperparameters, "stack", bool)
        if stacked:
            model = StackedRNN(
                cell_name=safe_load(hyperparameters, "cell_name", str),
                input_dim=safe_load(hyperparameters, "input_dim", int),
                state_dim=safe_load(hyperparameters, "state_dim", int),
                hidden_dim=safe_load(hyperparameters, "hidden_dim", int),
                output_dim=safe_load(hyperparameters, "output_dim", int),
                classification=safe_load(hyperparameters, "classification", bool),
                tanh_output=safe_load(hyperparameters, "tanh_output", bool),
                output_step=safe_load(hyperparameters, "output_step", int),
                drop_rate=safe_load(hyperparameters, "drop_rate", float),
                key=key,
            )
            state = eqx.nn.State(model)
        else:
            model = BasicRNN(
                cell_name=safe_load(hyperparameters, "cell_name", str),
                input_dim=safe_load(hyperparameters, "input_dim", int),
                state_dim=safe_load(hyperparameters, "state_dim", int),
                hidden_dim=safe_load(hyperparameters, "hidden_dim", int),
                output_dim=safe_load(hyperparameters, "output_dim", int),
            )
            state = None
        return model, state   
    elif model_name == "S5":
        model = S5(
            input_dim=safe_load(hyperparameters, "input_dim", int),
            state_dim=safe_load(hyperparameters, "state_dim", int),
            hidden_dim=safe_load(hyperparameters, "hidden_dim", int),
            output_dim=safe_load(hyperparameters, "output_dim", int),
            num_blocks=safe_load(hyperparameters, "num_blocks", int),
            classification=safe_load(hyperparameters, "classification", bool),
            tanh_output=safe_load(hyperparameters, "tanh_output", bool),
            output_step=safe_load(hyperparameters, "output_step", int),
            ssm_blocks=safe_load(hyperparameters, "ssm_blocks", int),
            C_init=safe_load(hyperparameters, "C_init", str),
            conj_sym=safe_load(hyperparameters, "conj_sym", bool),
            clip_eigs=safe_load(hyperparameters, "clip_eigs", bool),
            discretization=safe_load(hyperparameters, "discretization", str),
            dt_min=safe_load(hyperparameters, "dt_min", float),
            dt_max=safe_load(hyperparameters, "dt_max", float),
            step_rescale=safe_load(hyperparameters, "step_rescale", float),
            drop_rate=safe_load(hyperparameters, "drop_rate", float),
            key=key,
        )
        state = eqx.nn.State(model)
        return model, state
    elif model_name == "LRU":
        model = LRU(
            input_dim=safe_load(hyperparameters, "input_dim", int),
            state_dim=safe_load(hyperparameters, "state_dim", int),
            hidden_dim=safe_load(hyperparameters, "hidden_dim", int),
            output_dim=safe_load(hyperparameters, "output_dim", int),
            num_blocks=safe_load(hyperparameters, "num_blocks", int),
            classification=safe_load(hyperparameters, "classification", bool),
            tanh_output=safe_load(hyperparameters, "tanh_output", bool),
            output_step=safe_load(hyperparameters, "output_step", int),
            r_min=safe_load(hyperparameters, "r_min", float),
            theta_max=safe_load(hyperparameters, "theta_max", float),
            drop_rate=safe_load(hyperparameters, "drop_rate", float),
            key=key,
        )
        state = eqx.nn.State(model)
        return model, state
    elif model_name == "LinOSS":
        model = LinOSS(
            layer_name=safe_load(hyperparameters, "layer_name", str),
            input_dim=safe_load(hyperparameters, "input_dim", int),
            state_dim=safe_load(hyperparameters, "state_dim", int),
            hidden_dim=safe_load(hyperparameters, "hidden_dim", int),
            output_dim=safe_load(hyperparameters, "output_dim", int),
            num_blocks=safe_load(hyperparameters, "num_blocks", int),
            classification=safe_load(hyperparameters, "classification", bool),
            tanh_output=safe_load(hyperparameters, "tanh_output", bool),
            output_step=safe_load(hyperparameters, "output_step", int),
            initialization=safe_load(hyperparameters, "initialization", str),
            r_min=safe_load(hyperparameters, "r_min", float),
            r_max=safe_load(hyperparameters, "r_max", float),
            theta_min=safe_load(hyperparameters, "theta_min", float),
            theta_max=safe_load(hyperparameters, "theta_max", float),
            A_min=safe_load(hyperparameters, "A_min", float),
            A_max=safe_load(hyperparameters, "A_max", float),
            G_min=safe_load(hyperparameters, "G_min", float),
            G_max=safe_load(hyperparameters, "G_max", float),
            dt_std=safe_load(hyperparameters, "dt_std", float),
            drop_rate=safe_load(hyperparameters, "drop_rate", float),
            damping_mode=hyperparameters.get("damping_mode", "constant"),
            gate_type=hyperparameters.get("gate_type", "linear"),
            mult_min=float(hyperparameters.get("mult_min", 0.25)),
            mult_max=float(hyperparameters.get("mult_max", 4.0)),
            freq_aware_damping=bool(hyperparameters.get("freq_aware_damping", False)),
            zeta_min=float(hyperparameters.get("zeta_min", 0.0)),
            zeta_max=float(hyperparameters.get("zeta_max", 4.0)),
            gate_hidden_dim=hyperparameters.get("gate_hidden_dim", 64),
            gate_nonlinearity=hyperparameters.get("gate_nonlinearity", "gelu"),
            gate_use_energy=bool(hyperparameters.get("gate_use_energy", False)),
            use_block_deer=bool(hyperparameters.get("use_block_deer", False)),
            deer_num_iters=int(hyperparameters.get("deer_num_iters", 4)),
            deer_damping=float(hyperparameters.get("deer_damping", 0.0)),
            task_type=hyperparameters.get("task_type", "sequence"),
            key=key,
        )
        state = eqx.nn.State(model)
        return model, state
    else:
        raise ValueError(f"Unknown model name: {model_name}")