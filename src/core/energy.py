import logging
import threading
import time

import pynvml
import torch

from core.system import PYRAPL_AVAILABLE

logger = logging.getLogger(__name__)


class EnergyMonitor(threading.Thread):
    def __init__(self, device_type: str = "cuda", gpu_id: int = 0, sample_interval: float = 0.05, enable_rapl: bool = False):
        super().__init__()
        self.device_type = device_type
        self.gpu_id = gpu_id
        self.interval = sample_interval
        self.enable_rapl = enable_rapl
        self.stop_event = threading.Event()
        self.readings = []
        self.avg_power = 0.0
        self.nvml_handle = None
        self.cpu_meter = None
        self.daemon = True
        self._init_sensors()

    def _init_sensors(self):
        if self.device_type == "cuda" and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            except Exception as e:
                logger.warning(f"NVML Init failed: {e}")
                self.nvml_handle = None

        if self.device_type == "cpu" and PYRAPL_AVAILABLE and self.enable_rapl:
            try:
                import pyRAPL  # type: ignore
                pyRAPL.setup()  # type: ignore
                self.cpu_meter = pyRAPL.Measurement("cpu_meter")  # type: ignore
            except Exception as e:
                logger.warning(f"pyRAPL Init failed: {e}")
                self.cpu_meter = None

    def run(self):
        if self.cpu_meter:
            try:
                self.cpu_meter.begin()
            except Exception:
                pass

        while not self.stop_event.is_set():
            if self.device_type == "cuda" and self.nvml_handle:
                try:
                    p_mw = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                    self.readings.append(p_mw / 1000.0)
                except Exception:
                    self.readings.append(0.0)
            time.sleep(self.interval)

        if self.cpu_meter:
            try:
                self.cpu_meter.end()
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()
        self.join()

        if self.device_type == "cuda" and self.readings:
            self.avg_power = sum(self.readings) / len(self.readings)
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        elif self.device_type == "cpu" and self.cpu_meter and self.cpu_meter.result:
            res = self.cpu_meter.result
            if res.duration > 0 and res.pkg is not None and len(res.pkg) > 0:
                self.avg_power = (res.pkg[0] / 1e6) / (res.duration / 1e6)
            else:
                self.avg_power = 0.0

    def get_avg_power(self) -> float:
        return float(self.avg_power) if self.avg_power else 0.0
