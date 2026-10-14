import logging
import threading
import time
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd

from dss.common.logging_setup import setup_logger
from dss.database.connection import get_latest_step, save_dataset_to_postgresql

logger = setup_logger("dss.stream")


class SpectrumStreamWorker:
    """
    Background worker that continuously simulates and streams live 6G spectrum
    observations and MNO traffic demands into PostgreSQL.
    """

    def __init__(self, interval_seconds: float = 3.0, seed: int = 42):
        self.interval_seconds = interval_seconds
        self.rng = np.random.default_rng(seed)
        self.current_step = get_latest_step()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pu_states = self.rng.choice([0, 1], size=16, p=[0.6, 0.4])

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background streaming worker thread if not already running."""
        if self.is_running():
            logger.info("Streaming worker is already running at step #%d", self.current_step)
            return

        self._stop_event.clear()
        self.current_step = get_latest_step()
        self._thread = threading.Thread(target=self._run_loop, name="SpectrumStreamThread", daemon=True)
        self._thread.start()
        logger.info("Spectrum streaming worker started from step #%d (interval: %.1fs)", self.current_step + 1, self.interval_seconds)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the streaming worker to stop and wait for completion."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("Spectrum streaming worker stopped at step #%d", self.current_step)

    def _simulate_step(self, step_idx: int, ts_iso: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate realistic physical sensing parameters and multi-operator demands."""
        n_channels = 16
        n_mno = 4

        # 1. PU state Markov transition: ~15% chance to toggle state
        toggle_mask = self.rng.random(size=n_channels) < 0.15
        self._pu_states = np.where(toggle_mask, 1 - self._pu_states, self._pu_states)

        noise_est = -95.0 + self.rng.normal(0, 1.2, size=n_channels)
        interference = -102.0 + self.rng.normal(0, 1.8, size=n_channels)
        rx_power = np.where(
            self._pu_states == 1,
            -66.0 + self.rng.normal(0, 2.5, size=n_channels),
            noise_est + self.rng.uniform(1.0, 3.5, size=n_channels)
        )
        snr_est = rx_power - noise_est

        channel_df = pd.DataFrame({
            "step_idx": [step_idx] * n_channels,
            "ts": [ts_iso] * n_channels,
            "channel_id": list(range(n_channels)),
            "pu_state": self._pu_states.astype(np.int8),
            "noise_true_dbm": noise_est.astype(np.float32),
            "noise_est_dbm": noise_est.astype(np.float32),
            "interference_dbm": interference.astype(np.float32),
            "rx_power_dbm": rx_power.astype(np.float32),
            "snr_est_db": snr_est.astype(np.float32),
        })

        # 2. Dynamic MNO demands (4 licensed operators)
        load_multipliers = self.rng.uniform(0.4, 1.8, size=n_mno)
        base_subscribers = self.rng.integers(60, 380, size=n_mno)
        active_users = (base_subscribers * load_multipliers).astype(np.int32)
        traffic_demand = active_users * 1.5
        requested_bw = traffic_demand / 3.2

        mno_df = pd.DataFrame({
            "step_idx": [step_idx] * n_mno,
            "ts": [ts_iso] * n_mno,
            "mno_id": list(range(n_mno)),
            "active_users": active_users,
            "traffic_demand_mbps": traffic_demand.astype(np.float32),
            "requested_bw_mhz": requested_bw.astype(np.float32),
        })

        return channel_df, mno_df

    def _run_loop(self) -> None:
        """Main generator loop executing every interval."""
        while not self._stop_event.is_set():
            try:
                time.sleep(self.interval_seconds)
                if self._stop_event.is_set():
                    break

                self.current_step += 1
                step_idx = self.current_step
                ts_iso = datetime.now().isoformat()

                channel_df, mno_df = self._simulate_step(step_idx, ts_iso)
                save_dataset_to_postgresql(channel_df, mno_df)
                logger.info("Persisted Step #%d records to PostgreSQL [16 channels, 4 MNOs]", step_idx)

            except Exception as err:
                logger.error("Error during streaming step #%d: %s", self.current_step, err, exc_info=True)


_GLOBAL_STREAM_WORKER: Optional[SpectrumStreamWorker] = None


def get_stream_worker(interval_seconds: float = 3.0) -> SpectrumStreamWorker:
    """Return the global streaming worker instance."""
    global _GLOBAL_STREAM_WORKER
    if _GLOBAL_STREAM_WORKER is None:
        _GLOBAL_STREAM_WORKER = SpectrumStreamWorker(interval_seconds=interval_seconds)
    return _GLOBAL_STREAM_WORKER
