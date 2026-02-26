"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        VOXCORE — input/echo_cancel.py                           ║
║        ACOUSTIC ECHO CANCELLATION — WebRTC-INSPIRED ADAPTIVE FILTER            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Removes acoustic echo (AI speech leaking from speakers into the microphone)
    using a Partitioned Block Frequency Domain Adaptive Filter (PBFDAF).

    This is the same algorithm family used by WebRTC, Zoom, Discord, and
    Google Meet. It dynamically learns the speaker→mic transfer function
    (room impulse response) and subtracts the estimated echo from the mic
    signal in real-time.

    Unlike manual echo EMA hacks, this performs:
      • Adaptive filtering  — learns the room acoustics automatically
      • Delay estimation    — handles speaker→mic latency via delay line
      • Non-linear suppression — double-talk detection prevents corruption
      • No manual tuning    — converges within 1–2 seconds of playback

USAGE:
    aec = StreamingAEC(frame_size=480, num_blocks=12)

    # In audio callback (every 30ms):
    clean = aec.process(mic_frame, ref_frame)
    # `clean` contains user speech without echo

═══════════════════════════════════════════════════════════════════════════════════
"""

import logging

import numpy as np

logger = logging.getLogger("AEC")


# ─────────────────────────────────────────────────────────────────────────────────
# Streaming AEC — Partitioned Block Frequency Domain Adaptive Filter
# ─────────────────────────────────────────────────────────────────────────────────


class StreamingAEC:
    """Partitioned Block Frequency Domain Adaptive Filter for echo cancellation.

    The filter models the speaker→mic impulse response as a sum of
    frequency-domain partitions. Each partition covers one frame's worth
    of delay, so ``num_blocks × frame_size / sample_rate`` seconds of echo
    tail are modelled.

    For 16 kHz audio with 480-sample frames (30 ms) and 12 blocks,
    this models 360 ms of echo tail — sufficient for typical laptop/desktop
    environments.

    Stability measures (prevents divergence seen in v1):
      • High regularisation delta (1e-2) — prevents huge steps on quiet frames
      • Weight leakage (0.9995/step) — prevents unbounded weight growth
      • Output clipping to [-1, 1] — prevents overflow feedback
      • Divergence auto-reset — if output RMS > 2× input RMS, weights reset
      • Skip updates when reference is silent — nothing to learn from silence

    Args:
        frame_size:   Samples per frame (480 = 30 ms at 16 kHz).
        num_blocks:   Number of filter partitions.
        mu:           NLMS step size (0–1).  0.3 is safe for most rooms.
        delta:        Regularisation constant.  Must be large enough to
                      prevent div-by-near-zero on quiet frames.
        power_alpha:  Smoothing factor for reference PSD estimate (0–1).
        leak:         Weight leakage factor per frame (0–1).  Keeps weights
                      bounded.  0.9995 = slow decay ≈ 10s time constant.
    """

    def __init__(
        self,
        frame_size: int = 480,
        num_blocks: int = 12,
        mu: float = 0.3,
        delta: float = 1e-2,
        power_alpha: float = 0.90,
        leak: float = 0.9995,
    ) -> None:
        self.N = frame_size
        self.K = num_blocks
        self.mu = mu
        self.delta = delta
        self.power_alpha = power_alpha
        self.leak = leak

        # FFT size = 2× frame for overlap-save method
        self.fft_size = 2 * frame_size
        self.freq_bins = self.fft_size // 2 + 1

        # ── Adaptive filter state ──────────────────────────────────────────
        # W[k] — frequency-domain filter weights per partition
        self.W = np.zeros((num_blocks, self.freq_bins), dtype=np.complex128)

        # X[k] — frequency-domain reference delay line (newest at [0])
        self.X = np.zeros((num_blocks, self.freq_bins), dtype=np.complex128)

        # Power spectral density estimate for NLMS normalisation
        self.Pxx = np.full(self.freq_bins, delta, dtype=np.float64)

        # Previous reference frame for overlap-save windowing
        self.prev_ref = np.zeros(frame_size, dtype=np.float32)

        # ── Metrics ────────────────────────────────────────────────────────
        self.echo_return_loss_db: float = 0.0   # ERLE in dB (positive = good)
        self._frame_count: int = 0
        self._converged: bool = False
        self._divergence_count: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # Core processing
    # ─────────────────────────────────────────────────────────────────────

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Process one frame of audio through the echo canceller.

        Args:
            mic:  Microphone input, shape ``(N,)``, float32, range [-1, 1].
            ref:  Reference (playback) signal, shape ``(N,)``, float32.
                  Pass zeros when nothing is being played.

        Returns:
            Echo-cancelled signal, shape ``(N,)``, float32, clipped to [-1, 1].
            When no playback is happening, returns *mic* unchanged.
        """
        N = self.N

        # ── Quick-exit: if reference is silent, pass mic through unchanged ──
        ref_energy = float(np.sum(ref.astype(np.float64) ** 2))
        if ref_energy < 1e-10:
            self.prev_ref = ref.copy()
            # Still shift delay line so timing stays correct
            x_full = np.concatenate([self.prev_ref, ref]).astype(np.float64)
            X_curr = np.fft.rfft(x_full)
            self.X = np.roll(self.X, 1, axis=0)
            self.X[0] = X_curr
            return mic.copy()

        # ── 1. Overlap-save analysis of reference ──────────────────────────
        x_full = np.concatenate([self.prev_ref, ref]).astype(np.float64)
        X_curr = np.fft.rfft(x_full)

        # Shift delay line: newest → [0], oldest → [K-1]
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = X_curr

        # ── 2. Estimate echo: Y = Σ W[k]·X[k] ────────────────────────────
        Y = np.sum(self.W * self.X, axis=0)
        y_full = np.fft.irfft(Y, self.fft_size)
        echo_est = y_full[N:]  # overlap-save: valid region is last half

        # ── 3. Error = mic − echo estimate ─────────────────────────────────
        mic_f64 = mic.astype(np.float64)
        error = mic_f64 - echo_est

        # ── STABILITY: Clip error to valid audio range ─────────────────────
        error = np.clip(error, -1.0, 1.0)

        # ── STABILITY: Divergence detection ────────────────────────────────
        mic_energy = float(np.sum(mic_f64 ** 2))
        err_energy = float(np.sum(error ** 2))

        if err_energy > mic_energy * 4.0 and mic_energy > 1e-8:
            # Output is louder than input — filter has diverged
            self._divergence_count += 1
            if self._divergence_count > 5:
                logger.warning(
                    "AEC divergence detected (err=%.4f > mic=%.4f) — resetting filter",
                    err_energy, mic_energy,
                )
                self.reset()
                return mic.copy()
            # Return raw mic while we monitor
            return mic.copy()
        else:
            self._divergence_count = 0

        # ── 4. Update reference PSD (smoothed) ─────────────────────────────
        P_inst = np.real(X_curr * np.conj(X_curr))  # |X|²
        self.Pxx = self.power_alpha * self.Pxx + (1.0 - self.power_alpha) * P_inst

        # ── 5. Double-talk detection ───────────────────────────────────────
        # When user is speaking simultaneously, reduce step size to prevent
        # the filter from trying to model user speech as echo.
        mu = self.mu
        dtd = mic_energy / (ref_energy + 1e-10)
        if dtd > 5.0:
            mu = 0.0        # user voice dominant → freeze filter
        elif dtd > 2.5:
            mu *= 0.1       # probably double-talk → slow down
        elif dtd > 1.5:
            mu *= 0.3       # mild double-talk → moderate

        # ── 6. Constrained NLMS update ─────────────────────────────────────
        if mu > 0:
            # Weight leakage — prevents unbounded weight growth
            self.W *= self.leak

            # Build zero-padded error analysis window
            e_full = np.concatenate([np.zeros(N, dtype=np.float64), error])
            E = np.fft.rfft(e_full)

            # Normalisation denominator (shared across partitions)
            norm = self.Pxx + self.delta

            for k in range(self.K):
                # Raw gradient = E · conj(X[k])
                raw_grad = E * np.conj(self.X[k])

                # Constrain: project to time domain, zero last half, back
                # to freq domain.  This prevents the filter from learning
                # non-causal/circular components (overlap-save requirement).
                g_time = np.fft.irfft(raw_grad, self.fft_size)
                g_time[N:] = 0.0
                grad = np.fft.rfft(g_time)

                # Normalised step
                self.W[k] += mu * grad / norm

        # ── 7. Metrics ────────────────────────────────────────────────────
        if mic_energy > 1e-10 and err_energy < mic_energy:
            self.echo_return_loss_db = 10.0 * np.log10(
                mic_energy / (err_energy + 1e-10)
            )
        else:
            self.echo_return_loss_db = 0.0

        self._frame_count += 1
        if not self._converged and self._frame_count > 50 and self.echo_return_loss_db > 3.0:
            self._converged = True
            logger.info(
                "AEC converged after %d frames (ERLE=%.1f dB)",
                self._frame_count,
                self.echo_return_loss_db,
            )

        # Save for next overlap-save
        self.prev_ref = ref.copy()

        return error.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all filter state.  Call when environment changes drastically
        (e.g. headphones plugged in, volume changes by >10 dB)."""
        self.W[:] = 0
        self.X[:] = 0
        self.Pxx[:] = self.delta
        self.prev_ref[:] = 0
        self.echo_return_loss_db = 0.0
        self._frame_count = 0
        self._converged = False
        self._divergence_count = 0
        logger.info("AEC filter reset")

    @property
    def converged(self) -> bool:
        """Whether the filter has converged to a useful estimate."""
        return self._converged


# ─────────────────────────────────────────────────────────────────────────────────
# Reference Ring Buffer — SPSC lock-free for cross-thread audio
# ─────────────────────────────────────────────────────────────────────────────────


class RefRingBuffer:
    """Single-producer single-consumer ring buffer backed by a numpy array.

    Thread-safe for **one** writer (asyncio/playback thread) and **one**
    reader (PortAudio C callback thread) via CPython's GIL guaranteeing
    atomic integer assignment.

    The writer is ``AudioPlayer`` (asyncio thread, calling
    ``MicStream.set_playback_reference``).  The reader is
    ``MicStream._audio_callback`` (PortAudio C thread).

    Args:
        capacity:  Maximum number of float32 samples to store.
                   Default 32 000 ≈ 2 seconds at 16 kHz.
    """

    def __init__(self, capacity: int = 32000) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._wp: int = 0   # only modified by producer (write)
        self._rp: int = 0   # only modified by consumer (read)

    def write(self, data: np.ndarray) -> None:
        """Append samples.  Called from asyncio/playback thread."""
        n = len(data)
        if n == 0:
            return
        # If buffer would overflow, advance read pointer (drop oldest)
        avail_space = self._cap - (self._wp - self._rp)
        if n > avail_space:
            self._rp += n - avail_space

        wp = self._wp % self._cap
        if wp + n <= self._cap:
            self._buf[wp:wp + n] = data
        else:
            first = self._cap - wp
            self._buf[wp:] = data[:first]
            self._buf[:n - first] = data[first:]
        self._wp += n

    def read(self, n: int) -> np.ndarray:
        """Read and consume up to *n* samples.  Called from C callback thread.

        Returns exactly *n* samples.  If fewer are available, the remainder
        is zero-padded (silence — AEC treats this as "not playing").
        """
        avail = self._wp - self._rp
        got = min(n, max(0, avail))
        out = np.zeros(n, dtype=np.float32)

        if got > 0:
            rp = self._rp % self._cap
            if rp + got <= self._cap:
                out[:got] = self._buf[rp:rp + got]
            else:
                first = self._cap - rp
                out[:first] = self._buf[rp:]
                out[first:got] = self._buf[:got - first]
            self._rp += got

        return out

    @property
    def available(self) -> int:
        """Number of samples currently available to read."""
        return max(0, self._wp - self._rp)
