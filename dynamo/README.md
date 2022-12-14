# torchdynamo/torchinductor benchmark

## Install

Create a new virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Get the latest torch version with torchdynamo and inductor

```bash
pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
```

```bash
pip install -r requirements.txt
```

## Run tests

```bash
python3 test_grayscale.py
```

## Using the runner

### Setup a new benchmark

For this we need to add a python file which will have the `kornia` and `opencv` pair operations under the names: `kornia_op` and `opencv_op`.

Normally we for the `kornia_op` is just an alias to the op. The `opencv_op` normally will need a hand made function to be able to handle batch operations and also to use the same arguments name (aka translate) than `kornia_op`.

To maintain the sanity of this repository, we can maintain the same dir structure as kornia (where now the top level dir is the [config/](./config/)). The name of the python file under this dir must have a name that refers to the operation itself.

### Config the benchmark run

This is done a YAML file as example the [bench_config.yaml](./bench_config.yaml). This file is where the benchmark experiments are defined. First, we have a `global` level, and subsequently the config for each operation itself.

- The global level needs to have (as sublevels):
  1. `batch_sizes` a list of batch_sizes (integer) wanted to be tested in the benchmarks.
  1. `resolutions` a list of resolutions (integer) to test, this will generate an input with `ones` with shape (res, res)
  1. `threads` a list of the quantity of threads (integer) in the `benchmark.Timer`.
  1. `import_from` a string with the name of the top level directory with the desired benchmarks. Here, is the  [config/](./config/) directory.

- Subsequently we can add the operations wanted to be bench with their specific arguments.
  - The name of the operation should match with the relative path under the top level directory. For example, to config a `dilation` bench, which is in [config/geometry/transform/rescale.py](./config/geometry/transform/rescale.py) we will add the name of op as `geometry.transform.rescale`.
  - Here we can overwrite the `global` configs if necessary.
  - inside of the level of the operation we can add the arguments which will be dispatch to the operation itself.
    - Normal cases we can pass direct the desired argument, as example for rescale we want to pass the `factor` argument. Then we create a list of cases (for rescale a list of integer) to be done the benchmark.
    - Unique cases, we use constructors as:
        - `ones` which will create a torch tensor or a numpy array (filled with ones) with the shape passed as argument. Example, we have an argument `fill`, the config will look like
        ```yaml
        ...
            fill:
                ones:
                    [<the shape desired>]
        ```
        - TODO: add others operators as needed
          - `eye` will be necessary for the `geometric.transform.warp_affine`
        - Why have this unique cases? To be able to construct these arguments outside of the operation which is being performed the benchmark
    - If the operation doesn't need any other argument then the `input` you can use the argument `no_arg: True` at the operation config.

So the config file will look like:
```yaml
global:
  batch_sizes: [1, 9]
  resolutions: [128, 512]
  threads: [1, 8]
  import_from: 'config'

geometry.transform.rescale:
    factor:
      [
        [0.5, 0.5],
        [1.5, 1.5]
      ]
```

The strategy of the run will be a product between all configs -- this don't work for special arguments as `ones` now.

### Running the benchmark with the runner
With all the experiments desired configured in the `YAML` file you can run the runner with:
```bash
$ python runner.py
```

The arguments of the runner can be checked with the `--help` argument (running with `$python runner.py --help`). Some of the arguments are:
- `--config-filename` to define the `YAML` config file, by default the runner will look for `./bench_config.yaml`.
- `--verbose` to turn on the verbose mode of the dynamo. Also, have `--debug` to set logger to debug level.
