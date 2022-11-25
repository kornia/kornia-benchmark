import argparse
import logging
import pickle
import sys
from datetime import datetime
from itertools import product
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import numpy as np
import torch
import torch._dynamo as dynamo
import torch.utils.benchmark as benchmark
import yaml
from kornia.core import Tensor
from yaml.loader import SafeLoader


torch.set_float32_matmul_precision('high')
torch_dynamo_optimize = dynamo.optimize('inductor')


def create_inputs(
        bs: Optional[int],
        res: int,
        out_t: str,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
        RGB: bool = True,
) -> Union[Tensor, np.ndarray]:

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
        configs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int, int, int]:
    for cfg in configs:
        for bs, res in product(cfg['batch_sizes'], cfg['resolutions']):
            yield cfg, bs, res


def _iter_op_device() -> Tuple[str, str, str, Tuple[Any]]:
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
        module: str,
        operator: str,
        x: Union[Tensor, np.ndarray],
        optimizer: Any,
        **kwargs: Dict[str, Any]
) -> bool:
    try:
        module = __import__(module, fromlist=[None])
        op = getattr(module, operator)
        if optimizer:
            optimizer(op)(x, **kwargs)
        else:
            op(x, **kwargs)
        return True
    except Exception as err:
        del err
        return False


def _unpack_config_or_load_global(
        config_name: str,
        data: Dict[str, Any],
        global_data: Dict[str, Any],
) -> Any:
    if config_name in data:
        return data[config_name]

    return global_data[config_name]


def dict_product(d):
    # Same as itertools.product but between dict values
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


def load_config(filename: str) -> List[Dict[str, Any]]:
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
                cn: cv for cn, cv in v.items() if cv not in DEFAULT_CONFIGS
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


def run(
        configs: List[Dict[str, Any]],
        output_filename: str,
) -> int:
    with open(output_filename, 'wb') as fp:
        for operator, input_type, device, optimize in _iter_op_device():
            _opt_name, _opt_txt, _opt = optimize
            print(
                '-'*79,
                '\n\033[1;33m'
                f'-> Benchmarking {operator} at {device} {_opt_txt}'
                '\033[0;0m',
            )

            for cfg, bs, res in _iter_cfg(configs):
                x = create_inputs(
                    bs, res, input_type,
                    device=torch.device(device),
                )

                kwargs = cfg['kwargs']

                module_name = cfg['module']
                import_from = f'{cfg["import_from"]}.{module_name}'

                print(
                    f'\t->Module: {module_name} | Batch size={bs}, '
                    f'resolution={res}, args={kwargs}',
                )

                _args_values_str = ', '.join(str(v) for v in kwargs.values())
                sub_label = f'[{bs}, {res}, {_args_values_str}]'

                if _check_run(import_from, operator, x, _opt, **kwargs):
                    for num_threads in cfg['threads']:
                        print(
                            '\t\t-> benchmarking with '
                            f'num_threads={num_threads}...',
                        )

                        desc = f'{_opt_name}{operator.split("_")[0]}_{device}'
                        stmt = f'{operator}(input, **{kwargs})'
                        setup = f'from {import_from} import {operator}'

                        bench_out = benchmark.Timer(
                            stmt=stmt,
                            setup=setup,
                            globals={'input': x, **kwargs},
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='Performance kornia CLI',
        usage='python -m runner.py <command> [<args>]',
    )

    parser.add_argument(
        '--config-filename',
        default='ex_config_run.yaml',
        help='Filename for the YAML config for the runner',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
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
        torch._dynamo.config.log_level = logging.DEBUG

    return run(
        configs,
        output_filename=args.output_filename,
    )


if __name__ == '__main__':
    raise SystemExit(main())
