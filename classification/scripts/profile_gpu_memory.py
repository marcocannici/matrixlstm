import os
import time
import torch
import threading
import traceback

from tqdm import tqdm
import configargparse as argparse
from libs.trainer import add_train_params
from libs.arg_types import arg_boolean, arg_tuple
from libs.readers.transforms import add_transforms_params
from models.profiler_matrixlstm import MatrixProfiler, Statistics


class GpuMonitor(threading.Thread):

    def __init__(self, device=None, *args, **kwargs):
        super(GpuMonitor, self).__init__(*args, **kwargs)

        self.device = device
        self._gpu_mem_peak = 0
        self._stop_event = threading.Event()

    @property
    def gpu_mem_peak(self):
        return self._gpu_mem_peak # bytes

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self.stopped:
            mem = torch.cuda.memory_allocated(device=self.device)
            self._gpu_mem_peak = max(self._gpu_mem_peak, mem)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def get_params():
    parser = argparse.ArgumentParser()
    parser = add_train_params(parser)
    parser = add_transforms_params(parser)

    parser.add_argument('--output_file', type=str)
    parser.add_argument('--input_height', type=int, default=224)
    parser.add_argument('--input_width', type=int, default=224)
    parser.add_argument('--lstm_type', type=str, default='LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=3)
    parser.add_argument('--region_shape', type=arg_tuple(int), default=(1, 1))
    parser.add_argument('--region_stride', type=arg_tuple(int), default=(1, 1))
    parser.add_argument('--add_coords_feature', type=arg_boolean, default=False)
    parser.add_argument('--add_time_feature_mode', type=str, default='delay_norm')
    parser.add_argument('--normalize_relative', type=arg_boolean, default=True)
    parser.add_argument('--keep_most_recent', type=arg_boolean, default=False)
    parser.add_argument('--frame_intervals', type=int, default=1)
    parser.add_argument('--frame_intervals_mode', type=str, default=None)
    parser.add_argument('--training', type=arg_boolean, default=True)
    parser.add_argument('--density', type=float, default=1.0)

    args, _ = parser.parse_known_args()
    return args


def trial(params, lstm_type, batch_size, hidden_size, time_size, kdim=1, density=1.0, training=False, save=True):

    torch.manual_seed(42)

    bw_statistics = Statistics()
    with GpuMonitor() as monitor:
        try:
        # assert torch.cuda.memory_allocated() == 0

            net = MatrixProfiler((params.input_height, params.input_width),
                                 hidden_size, [kdim, kdim], [1, 1],
                                 True if lstm_type == "LSTM" and kdim > 1 else params.add_coords_feature,
                                 params.add_time_feature_mode,
                                 params.normalize_relative, lstm_type, params.keep_most_recent,
                                 params.frame_intervals, params.frame_intervals_mode,
                                 params.lstm_num_layers)
            net.to(device)

            if training:
                net.train()
            else:
                net.eval()
            net_space = torch.cuda.memory_allocated()

            # If sparsity mode is enabled, time_size is considered per pixel
            nactive_pixels = int(density * params.input_height * params.input_width)
            xx, yy = torch.meshgrid(torch.arange(params.input_width), torch.arange(params.input_height))
            all_pixels = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(device)

            batch_events = torch.zeros([batch_size, nactive_pixels*time_size, 4], device=device)
            batch_lengths = torch.tensor([nactive_pixels*time_size] * batch_size, device=device).long()
            # Select nactive_pixels different pixels for each sample
            for i in range(batch_size):
                active = all_pixels[torch.randperm(all_pixels.shape[0])[:nactive_pixels]]  # nactive,2
                active = active[None].repeat(time_size, 1, 1).reshape(-1, 2)  # timesize*nactive, 2
                active = active[torch.randperm(active.shape[0]).to(device)]  # timesize*nactive, 2
                batch_events[i, :, 0] = active[:, 0]  # x
                batch_events[i, :, 1] = active[:, 1]  # y
                batch_events[i, :, 2] = torch.arange(active.shape[0], device=device)  # ts
                batch_events[i, :, 3] = torch.randint(0, 2, size=[active.shape[0]], device=device)  # p

            input_space = torch.cuda.memory_allocated() - net_space

            with GpuMonitor() as fw_monitor:
                out = net.forward(batch_events, batch_lengths, burned_in=True)

            if training:
                with GpuMonitor() as bw_monitor:
                    bw_start = time.time()
                    out.mean().backward()
                    bw_time = time.time() - bw_start
                bw_statistics.update(bw_time, batch_lengths)

            # Remove input gpu allocation
            del active, batch_events, batch_lengths, out
            torch.cuda.empty_cache()

            processing_space = monitor.gpu_mem_peak - net_space - input_space
            fw_processing_space = fw_monitor.gpu_mem_peak - net_space - input_space
            fw_mean_time, fw_kevents_s, n_samples = net.get_statistics()

            bw_processing_space, bw_mean_time = 0, 0
            if training:
                bw_processing_space = bw_monitor.gpu_mem_peak - net_space - input_space
                bw_mean_time, _, _ = bw_statistics.get()

            del net

            if save:
                with open(params.output_file, 'a') as f:
                    f.write(f"{lstm_type};{batch_size};{time_size};{density};{params.frame_intervals};"
                            f"{hidden_size};{kdim};{net_space};{input_space};{processing_space};"
                            f"{monitor.gpu_mem_peak};{fw_kevents_s};{fw_mean_time};{fw_processing_space};"
                            f"{bw_mean_time};{bw_processing_space}\n".replace(".", ","))
        except:
            return True
        return False


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params = get_params()
    parentdir = os.path.dirname(params.output_file)
    if parentdir:
        os.makedirs(parentdir, exist_ok=True)

    # Write headers
    with open(params.output_file, 'a') as f:
        f.write("lstm_type;batch_size;time_size;density;n_intervals;hidden_size;"
                "kernel_size;net_space;input_space;all_processing_space;all_gpu_peak;"
                "fw_kevents_s;fw_mean_time;fw_processing_space;"
                "bw_mean_time;bw_processing_space\n")

    assert torch.cuda.memory_allocated() == 0
    # Burn-in
    print("Burn-in")
    for _ in tqdm(range(20)):
        _ = trial(params, "LSTM", 4, 4, 1000, 1, density=0.1, save=False)

    for lstm_type in ["ConvLSTM", "LSTM"]:
        print(f"Analyzing {lstm_type}")
        pbar = tqdm()

        time_size = 100 if params.density is None else 1
        time_crashed = False
        while not time_crashed:

            batch_size = 1
            batch_crashed = False
            while not batch_crashed:

                hidden_size = 1
                hidden_crashed = False

                while not hidden_crashed:
                    if not params.training:
                        with torch.no_grad():
                            hidden_crashed = trial(params, lstm_type, batch_size, hidden_size, time_size,
                                                   density=params.density)
                    else:
                        hidden_crashed = trial(params, lstm_type, batch_size, hidden_size, time_size,
                                               density=params.density, training=True)

                    pbar.update(1)
                    hidden_size *= 2
                    hidden_crashed = hidden_crashed or hidden_size > 128

                # We stop incrementing batch if it saturates
                # the gpu even for hidden_size = 1
                batch_size *= 2
                batch_crashed = batch_size > 512 or hidden_size == 2

            # We stop incrementing time if it saturates
            # the gpu even for batch_size = 1
            time_size *= 10
            time_crashed = batch_size == 2
