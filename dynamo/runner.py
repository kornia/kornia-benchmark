from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime
from functools import partial
from itertools import product
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark
import yaml
from kornia.core import Tensor
from plot import graphs_from_results
from yaml.loader import SafeLoader


torch.set_float32_matmul_precision('high')
torch_dynamo_optimize = dynamo.optimize('inductor')


def not_implemented_op(*args, **kwargs) -> None:
    raise NotImplementedError(
        'Some error when setting the operation occurs.'
        'Default not implemented function called!',
    )


def _load_operator(
    module: str, operator: str, optimizer: Any,
) -> Callable[..., Any]:
    module = __import__(module, fromlist=[None])
    op = getattr(module, operator)
    if optimizer:
        return optimizer(op)

    return op


def set_op_to_bench(module: str, operator: str, optimizer: Any) -> None:
    global op_to_bench
    op_to_bench = not_implemented_op
    op_to_bench = _load_operator(module, operator, optimizer)


def create_inputs(
        bs: int | None,
        res: int,
        out_t: str,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
        RGB: bool = True,
) -> Tensor | np.ndarray:

    if RGB:
        x_tensor = torch.ones((3, res, res), dtype=dtype, device=device)
        if bs is not None:
            x_tensor = x_tensor[None].repeat(bs, 1, 1, 1)
    else:
        # TODO
        raise NotImplementedError

    if out_t == 'tensor':
        return x_tensor

    x_array: np.ndarray = x_tensor.detach().cpu().numpy()
    return x_array


def _iter_cfg(
        configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, int, int]:
    for cfg in configs:
        for bs, res in product(cfg['batch_sizes'], cfg['resolutions']):
            yield cfg, bs, res


def _iter_op_device() -> tuple[str, str, str, tuple[Any]]:
    # TODO: Maybe automate this?
    _iters = [
        ('kornia_op', 'tensor', 'cpu', None),
        ('kornia_op', 'tensor', 'cuda', None),
        ('kornia_op', 'tensor', 'cpu', ('dynamo', torch_dynamo_optimize)),
        ('kornia_op', 'tensor', 'cuda', ('dynamo', torch_dynamo_optimize)),
        ('opencv_op', 'numpy', 'cpu', None),
    ]

    for operator, input_type, device, optimize in _iters:
        if optimize is None:
            _opt_name = ''
            _opt_txt = ''
            _opt = False
        else:
            _opt_name = optimize[0] + '_'
            _opt = optimize[1]
            _opt_txt = f'with optimizer {_opt_name}'

        yield operator, input_type, device, (_opt_name, _opt_txt, _opt)


def _check_run(
        verbose: bool,
        module: str,
        operator: str,
        x: Tensor | np.ndarray,
        optimizer: Any,
        **kwargs: dict[str, Any]
) -> bool:
    try:
        op = _load_operator(module, operator, optimizer)
        op(x, **kwargs)
        return True
    except Exception as err:
        if verbose:
            print(
                '\n\n\n', '-' * 79,
                '\033[1;31m',
                f'\t\tException on running {module}\n',
                err,
                '\033[0;0m',
                '\n\n\n', '-' * 79,
            )
        del err
        return False


def _unpack_config_or_load_global(
        config_name: str,
        data: dict[str, Any],
        global_data: dict[str, Any],
) -> Any:
    if config_name in data:
        return data[config_name]

    return global_data[config_name]


def create_ones(
        shape: tuple[int, ...],
        out_t: str,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
) -> Tensor | np.ndarray:
    x_tensor = torch.ones(shape, dtype=dtype, device=device)
    if out_t == 'tensor':
        return x_tensor

    x_array: np.ndarray = x_tensor.detach().cpu().numpy()
    return x_array


def _unpack_config(i):
    if isinstance(i, dict):
        if 'ones' in i:
            # TODO: figure out a way to declare multiple cases.
            # Ex config differents kernel sizes
            if isinstance(i['ones'], list) and len(i['ones']) >= 2:
                shape = tuple(int(x) for x in i['ones'])
            elif isinstance(i['ones'], list) and len(i['ones']) == 1:
                _d = int(i['ones'][0])
                shape = (_d, _d)
            return partial(create_ones, shape=shape)
        else:
            raise NotImplementedError
    return i


def dict_product(data):
    # Same as itertools.product but between dict values
    prod_cases = {k: v for k, v in data.items() if not callable(v)}
    others = {k: v for k, v in data.items() if callable(v)}

    for element in product(*prod_cases.values()):
        out_prod = dict(zip(prod_cases.keys(), element))
        yield {**out_prod, **others}


def load_config(filename: str) -> list[dict[str, Any]]:
    with open(filename) as f:
        data = yaml.load(f, Loader=SafeLoader)

    global_config = data['global']

    DEFAULT_CONFIGS = ['batch_sizes', 'resolutions', 'threads', 'import_from']

    config = [
        {
            'module': k,
            'kwargs': lc,
            ** {
                cn: _unpack_config_or_load_global(cn, v, global_config)
                for cn in DEFAULT_CONFIGS
            },
        }
        for k, v in data.items() if k != 'global'
        for lc in dict_product(
            {
                cn: _unpack_config(cv) for cn, cv in v.items()
                if cv not in DEFAULT_CONFIGS and cn != 'no_args'
            },
        )
    ]

    return config


def _unpick(filename: str) -> list[Any]:
    results = []
    with open(filename, 'rb') as fp:
        try:
            while True:
                results.append(pickle.load(fp))
        except EOFError:
            pass

    return results


def _build_kwargs(
        f_kwargs: dict[str, Any],
        out_t: str,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
):
    def _unpack_arg(arg,  out_t=out_t, dtype=dtype, device=device):
        if callable(arg):
            return arg(out_t=out_t, dtype=dtype, device=device)
        return arg

    return {
        k: _unpack_arg(v)
        for k, v in f_kwargs.items()
    }


def run(
        configs: list[dict[str, Any]],
        output_filename: str,
        verbose: bool,
) -> int:
    with open(output_filename, 'wb') as fp:
        for operator, input_type, device, optimize in _iter_op_device():
            optz_name, optimizer_txt, optimizer = optimize
            _op_dev_txt = (
                f'\033[1;33m {operator} at {device} {optimizer_txt}'
                '\033[0;0m'
            )
            print(
                '-'*79,
                f'\n-> Benchmarking{_op_dev_txt}',
            )

            for cfg, bs, res in _iter_cfg(configs):
                x = create_inputs(
                    bs, res, input_type,
                    device=torch.device(device),
                )

                kwargs = _build_kwargs(
                    cfg['kwargs'],
                    out_t=input_type,
                    device=torch.device(device),
                )

                module_name = cfg['module']
                module = f'{cfg["import_from"]}.{module_name}'

                _args_values_str = ', '.join(
                    str(tuple(v.shape)) if hasattr(v, 'shape')
                    else str(v)
                    for v in kwargs.values()
                )
                sub_label = f'[{bs}, {res}, {_args_values_str}]'

                print(
                    '\n\n\t', '-'*70, '\n'
                    f'\t->({_op_dev_txt}) Module: {module_name} |'
                    f'Batch size={bs}, '
                    f'resolution={res}, args={_args_values_str}',
                )

                if _check_run(
                        verbose, module, operator, x, optimizer, **kwargs
                ):
                    # Load the operation on the scope of this file and
                    # optimize if is the case
                    set_op_to_bench(module, operator, optimizer)

                    for num_threads in cfg['threads']:
                        print(
                            '\t\t-> benchmarking with '
                            f'num_threads={num_threads}...',
                        )

                        desc = f'{optz_name}{operator.split("_")[0]}_{device}'
                        setup = 'from __main__ import op_to_bench'
                        stmt = 'op_to_bench(input, **kwargs)'
                        values = {'input': x, 'kwargs': kwargs}

                        bench_out = benchmark.Timer(
                            stmt=stmt,
                            setup=setup,
                            globals=values,
                            num_threads=num_threads,
                            label=module_name,
                            sub_label=sub_label,
                            description=desc,
                        ).blocked_autorange(min_run_time=1)

                        print(
                            '\033[1;32m'
                            '\t\t-> Saving benchmark...'
                            '\033[0;0m',
                        )
                        pickle.dump(
                            bench_out,
                            fp,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                else:
                    print(
                        '\033[1;31m'
                        '\t\t-> Fail to run. Skipping benchmark...'
                        '\033[0;0m',
                    )

    ab_results = _unpick(output_filename)
    compare = benchmark.Compare(ab_results)
    compare.print()

    return 0


def generate_graphs(filename: str) -> int:
    # Load the results: will be a list of torch Measurement
    graphs_from_results(_unpick(filename))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='Performance kornia CLI',
        usage='python -m runner.py <command> [<args>]',
    )

    parser.add_argument(
        '--config-filename',
        default='bench_config.yaml',
        help='Filename for the YAML config for the runner',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--save-graphs',
        action='store_true',
        default=False,
        help=(
            'From the benchmark file, generate a graph for each operation'
            'comparing the benchmarks.'
        ),
    )

    _dt = datetime.strftime(datetime.utcnow(), '%Y%m%d_%H%M%S')
    parser.add_argument(
        '--output-filename',
        default=f'output-benchmark-{_dt}.pickle',
        help='Filename for the pickle output file',
    )

    args = parser.parse_args(argv)

    configs = load_config(args.config_filename)

    if args.verbose:
        torch._dynamo.config.verbose = True
        torch._dynamo.config.log_level = logging.INFO
    elif args.debug:
        torch._dynamo.config.verbose = True
        torch._dynamo.config.log_level = logging.DEBUG

    sig = run(
        configs,
        output_filename=args.output_filename,
        verbose=args.verbose or args.debug,
    )

    if args.save_graphs:
        sig = generate_graphs(filename=args.output_filename)

    return sig

if __name__ == '__main__':
    raise SystemExit(main())
