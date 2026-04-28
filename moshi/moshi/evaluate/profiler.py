# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler


class Profiler:
    def __init__(self, models: dict[str, torch.nn.Module]):
        self.flops_profilers = {name: FlopsProfiler(model) for name, model in models.items()}

    def start_profile(self) -> None:
        for _, flops_profiler in self.flops_profilers.items():
            flops_profiler.start_profile()

    def end_profile(self) -> dict[str, dict[str, int]]:
        def profile_output_to_int(rep) -> int:
            try:
                rep = eval(rep)
            except Exception:
                rep = int(round(float(str(rep).split("+")[-1])))
            return rep

        results = {}

        for name, flops_profiler in self.flops_profilers.items():
            flops = flops_profiler.get_total_flops()
            macs = flops_profiler.get_total_macs()
            params = flops_profiler.get_total_params()
            flops_profiler.end_profile()
            if not isinstance(flops, int):
                flops = profile_output_to_int(flops)
            if not isinstance(macs, int):
                macs = profile_output_to_int(macs)
            if not isinstance(params, int):
                params = profile_output_to_int(params)

            results[name] = {
                "flops": flops,
                "macs": macs,
                "params": params,
            }
        return results
