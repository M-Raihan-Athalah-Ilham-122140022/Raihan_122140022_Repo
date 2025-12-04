import cv2
import numpy as np
import mediapipe as mp
import time
from scipy import signal
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class EnhancedRealTimeRPPG:
    def __init__(self, video_source=0, buffer_size=150, fps=30):
        """
        Enhanced rPPG system dengan multiple improvements:
        - POS (Plane Orthogonal to Skin) method
        - Motion artifact detection
        - Adaptive filtering
        - Advanced visualization
        """
        # Inisialisasi Webcam
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Parameter Sinyal
        self.buffer_size = buffer_size
        self.fps = fps
        self.signal_buffer = deque(maxlen=buffer_size)
        self.rgb_buffer = deque(maxlen=buffer_size)  # Buffer untuk metode POS
        self.bpm_history = deque(maxlen=30)  # History BPM untuk smoothing
        self.current_bpm = 0
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # ROI Definitions - Multiple regions untuk robust detection
        # Pipi (paling vaskular)
        self.left_cheek = [116, 117, 118, 100, 126, 209, 192, 213, 147, 123]
        self.right_cheek = [345, 346, 347, 329, 355, 429, 416, 433, 376, 352]
        
        # Dahi (forehead) - backup ROI
        self.forehead = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Tracking variables untuk motion detection
        self.prev_roi_position = None
        self.motion_threshold = 5.0  # pixels
        self.is_motion_detected = False
        
        # Signal quality metrics
        self.signal_quality = 0.0
        self.snr = 0.0
        
        # Method selection
        self.use_pos_method = True  # Toggle untuk POS vs Green channel
        
        # Visualization setup
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup untuk advanced visualization"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 4))
        self.fig.patch.set_facecolor('#2e2e2e')
        
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            
        self.canvas = FigureCanvasAgg(self.fig)

    def get_roi_average(self, frame, landmarks, roi_indices):
        """Ekstraksi rata-rata RGB dari ROI tertentu"""
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Konversi landmark ke koordinat
        points = []
        for idx in roi_indices:
            pt = landmarks.landmark[idx]
            points.append((int(pt.x * w), int(pt.y * h)))
        
        if len(points) < 3:
            return None, None, None, None
            
        pts = np.array(points, np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        
        # Ekstraksi mean RGB
        mean_b, mean_g, mean_r, _ = cv2.mean(frame, mask=mask)
        
        # Hitung centroid untuk motion tracking
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
        else:
            centroid = None
        
        return mean_r, mean_g, mean_b, pts

    def detect_motion(self, current_centroid):
        """Deteksi gerakan berlebihan yang dapat menyebabkan artifact"""
        if self.prev_roi_position is None:
            self.prev_roi_position = current_centroid
            return False
            
        if current_centroid is None:
            return True
            
        distance = np.sqrt((current_centroid[0] - self.prev_roi_position[0])**2 + 
                          (current_centroid[1] - self.prev_roi_position[1])**2)
        
        self.prev_roi_position = current_centroid
        self.is_motion_detected = distance > self.motion_threshold
        
        return self.is_motion_detected

    def extract_pos_signal(self):
        """
        Plane-Orthogonal-to-Skin (POS) method
        Lebih robust terhadap motion artifact dibanding simple green channel
        Reference: Wang et al. (2017) - Algorithmic Principles of Remote PPG
        """
        if len(self.rgb_buffer) < 2:
            return 0
            
        # Convert buffer to numpy array
        C = np.array(self.rgb_buffer).T  # Shape: (3, N) -> [R, G, B]
        
        # Normalization
        mean_C = np.mean(C, axis=1, keepdims=True)
        C_normalized = C / (mean_C + 1e-6)
        
        # Build projection plane
        S = np.array([[0, 1, -1],
                      [-2, 1, 1]]) @ C_normalized
        
        # Calculate pulse signal
        h = S[0, :]
        
        # Standardization
        h = (h - np.mean(h)) / (np.std(h) + 1e-6)
        
        return h[-1] if len(h) > 0 else 0

    def adaptive_bandpass_filter(self, raw_signal, prev_bpm=None):
        """
        Adaptive bandpass filter yang menyesuaikan range berdasarkan BPM sebelumnya
        """
        if len(raw_signal) < self.buffer_size:
            return raw_signal
            
        # Detrending menggunakan polynomial fit
        x = np.arange(len(raw_signal))
        z = np.polyfit(x, raw_signal, 3)
        p = np.poly1d(z)
        detrended = raw_signal - p(x)
        
        # Adaptive filter range
        nyquist = 0.5 * self.fps
        
        if prev_bpm and 40 < prev_bpm < 180:
            # Narrow band around previous BPM
            center_freq = prev_bpm / 60.0
            low = max(0.5, center_freq - 0.3) / nyquist
            high = min(4.0, center_freq + 0.3) / nyquist
        else:
            # Wide band untuk initial detection
            low = 0.67 / nyquist
            high = 4.0 / nyquist
        
        # Butterworth bandpass filter (order 4 untuk better attenuation)
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, detrended)
        
        return filtered

    def calculate_signal_quality(self, filtered_signal):
        """
        Hitung kualitas sinyal menggunakan SNR (Signal-to-Noise Ratio)
        """
        if len(filtered_signal) < self.buffer_size:
            return 0.0, 0.0
            
        # FFT untuk mendapatkan spektrum
        fft_res = np.fft.rfft(filtered_signal)
        freqs = np.fft.rfftfreq(len(filtered_signal), 1.0/self.fps)
        power_spectrum = np.abs(fft_res)**2
        
        # Identifikasi signal band (0.67-4.0 Hz)
        signal_idx = np.where((freqs >= 0.67) & (freqs <= 4.0))[0]
        noise_idx = np.where((freqs > 4.0) & (freqs < self.fps/2))[0]
        
        if len(signal_idx) == 0 or len(noise_idx) == 0:
            return 0.0, 0.0
            
        signal_power = np.sum(power_spectrum[signal_idx])
        noise_power = np.sum(power_spectrum[noise_idx])
        
        # Calculate SNR in dB
        snr = 10 * np.log10((signal_power + 1e-6) / (noise_power + 1e-6))
        
        # Quality score (0-1)
        quality = np.clip(snr / 20.0, 0, 1)
        
        return quality, snr

    def calculate_bpm_fft(self, filtered_signal):
        """Estimasi BPM menggunakan FFT dengan peak detection"""
        if len(filtered_signal) < self.buffer_size:
            return 0
            
        # Windowing untuk mengurangi spectral leakage
        window = signal.windows.hann(len(filtered_signal))
        signal_windowed = filtered_signal * window
        
        # FFT
        fft_res = np.fft.rfft(signal_windowed)
        freqs = np.fft.rfftfreq(len(signal_windowed), 1.0/self.fps)
        power = np.abs(fft_res)**2
        
        # Valid frequency range (40-240 BPM)
        valid_idx = np.where((freqs >= 0.67) & (freqs <= 4.0))[0]
        
        if len(valid_idx) == 0:
            return 0
            
        valid_power = power[valid_idx]
        valid_freqs = freqs[valid_idx]
        
        # Find peaks menggunakan scipy
        peaks, properties = signal.find_peaks(valid_power, 
                                             height=np.max(valid_power)*0.3,
                                             distance=5)
        
        if len(peaks) == 0:
            # Fallback ke maximum
            peak_idx = np.argmax(valid_power)
        else:
            # Ambil peak tertinggi
            peak_idx = peaks[np.argmax(properties['peak_heights'])]
        
        dominant_freq = valid_freqs[peak_idx]
        bpm = dominant_freq * 60.0
        
        # Smoothing dengan history
        if 40 <= bpm <= 180:  # Valid heart rate range
            self.bpm_history.append(bpm)
            if len(self.bpm_history) > 5:
                # Weighted moving average
                weights = np.exp(np.linspace(-1, 0, len(self.bpm_history)))
                weights /= weights.sum()
                bpm = np.average(list(self.bpm_history), weights=weights)
        
        return bpm

    def create_advanced_plot(self, time_signal, freq_signal, freqs, power):
        """Generate advanced visualization dengan time & frequency domain"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Time domain plot
        self.ax1.set_title('Time Domain Signal', color='white', fontsize=10)
        self.ax1.set_xlabel('Samples', color='white', fontsize=8)
        self.ax1.set_ylabel('Amplitude', color='white', fontsize=8)
        
        if len(time_signal) > 0:
            x_time = np.arange(len(time_signal))
            self.ax1.plot(x_time, time_signal, color='#00ff41', linewidth=1.5)
            self.ax1.fill_between(x_time, time_signal, alpha=0.3, color='#00ff41')
            self.ax1.grid(True, alpha=0.2, color='white')
        
        # Frequency domain plot
        self.ax2.set_title('Frequency Domain (Power Spectrum)', color='white', fontsize=10)
        self.ax2.set_xlabel('Frequency (Hz)', color='white', fontsize=8)
        self.ax2.set_ylabel('Power', color='white', fontsize=8)
        
        if len(freqs) > 0 and len(power) > 0:
            # Plot hanya range yang valid
            valid_idx = np.where((freqs >= 0.5) & (freqs <= 4.5))[0]
            self.ax2.plot(freqs[valid_idx], power[valid_idx], 
                         color='#ff00ff', linewidth=1.5)
            self.ax2.fill_between(freqs[valid_idx], power[valid_idx], 
                                 alpha=0.3, color='#ff00ff')
            
            # Mark peak (current BPM)
            if self.current_bpm > 0:
                peak_freq = self.current_bpm / 60.0
                self.ax2.axvline(x=peak_freq, color='red', linestyle='--', 
                               linewidth=2, label=f'BPM: {self.current_bpm:.1f}')
                self.ax2.legend(loc='upper right', fontsize=8, 
                              facecolor='#2e2e2e', edgecolor='white')
            
            self.ax2.grid(True, alpha=0.2, color='white')
        
        self.fig.tight_layout()
        
        # Convert plot to image
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(self.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        
        return plot_img

    def draw_ui(self, frame):
        """Draw comprehensive UI overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # BPM Display (Large)
        bpm_color = (0, 255, 0) if 60 <= self.current_bpm <= 100 else (0, 165, 255)
        if self.current_bpm < 40 or self.current_bpm > 180:
            bpm_color = (0, 0, 255)
            
        cv2.putText(frame, f"BPM: {self.current_bpm:.1f}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 3)
        
        # Signal Quality
        quality_color = (0, int(255*self.signal_quality), int(255*(1-self.signal_quality)))
        cv2.putText(frame, f"Quality: {self.signal_quality*100:.0f}%", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        # SNR
        cv2.putText(frame, f"SNR: {self.snr:.1f} dB", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Method indicator
        method_text = "Method: POS" if self.use_pos_method else "Method: Green"
        cv2.putText(frame, method_text, (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Motion warning
        if self.is_motion_detected:
            cv2.putText(frame, "MOTION DETECTED!", (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Buffer status
        buffer_pct = (len(self.signal_buffer) / self.buffer_size) * 100
        cv2.rectangle(frame, (420, 20), (620, 40), (50, 50, 50), -1)
        cv2.rectangle(frame, (420, 20), (420 + int(200 * buffer_pct/100), 40), 
                     (0, 255, 0), -1)
        cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%", (425, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q': Quit | 'm': Toggle Method | 'r': Reset", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def run(self):
        """Main processing loop"""
        print("=" * 60)
        print("Enhanced Real-Time rPPG System")
        print("=" * 60)
        print("Features:")
        print("  - POS Method untuk robust signal extraction")
        print("  - Adaptive bandpass filtering")
        print("  - Motion artifact detection")
        print("  - Signal quality monitoring (SNR)")
        print("  - Real-time visualization (time & frequency domain)")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'm' - Toggle POS/Green method")
        print("  'r' - Reset buffers")
        print("=" * 60)
        
        prev_time = time.time()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror untuk user experience
            
            # Calculate FPS
            current_time = time.time()
            fps_real = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract dari multiple ROIs
                    roi_list = [self.left_cheek, self.right_cheek, self.forehead]
                    rgb_values = []
                    all_polys = []
                    centroid = None
                    
                    for roi in roi_list:
                        r, g, b, poly = self.get_roi_average(frame, face_landmarks, roi)
                        if r is not None:
                            rgb_values.append([r, g, b])
                            all_polys.append(poly)
                            if centroid is None and poly is not None:
                                M = cv2.moments(poly)
                                if M["m00"] != 0:
                                    centroid = (int(M["m10"] / M["m00"]), 
                                              int(M["m01"] / M["m00"]))
                    
                    if len(rgb_values) > 0:
                        # Average dari semua ROIs
                        avg_rgb = np.mean(rgb_values, axis=0)
                        
                        # Motion detection
                        self.detect_motion(centroid)
                        
                        # Extract signal berdasarkan method
                        if self.use_pos_method:
                            self.rgb_buffer.append(avg_rgb)
                            if len(self.rgb_buffer) >= 10:
                                signal_value = self.extract_pos_signal()
                            else:
                                signal_value = avg_rgb[1]  # Fallback ke green
                        else:
                            signal_value = avg_rgb[1]  # Green channel only
                        
                        self.signal_buffer.append(signal_value)
                        
                        # Process signal jika buffer cukup
                        if len(self.signal_buffer) >= self.buffer_size:
                            raw_signal = np.array(self.signal_buffer)
                            
                            # Adaptive filtering
                            prev_bpm = self.current_bpm if self.current_bpm > 0 else None
                            filtered = self.adaptive_bandpass_filter(raw_signal, prev_bpm)
                            
                            # Calculate BPM
                            self.current_bpm = self.calculate_bpm_fft(filtered)
                            
                            # Calculate signal quality
                            self.signal_quality, self.snr = self.calculate_signal_quality(filtered)
                            
                            # Generate visualization
                            fft_res = np.fft.rfft(filtered * signal.windows.hann(len(filtered)))
                            freqs = np.fft.rfftfreq(len(filtered), 1.0/self.fps)
                            power = np.abs(fft_res)**2
                            
                            plot_img = self.create_advanced_plot(filtered, filtered, freqs, power)
                            
                            # Overlay plot pada frame
                            plot_h, plot_w = plot_img.shape[:2]
                            frame_h, frame_w = frame.shape[:2]
                            
                            # Resize jika perlu
                            scale = min(frame_w * 0.4 / plot_w, frame_h * 0.5 / plot_h)
                            new_w, new_h = int(plot_w * scale), int(plot_h * scale)
                            plot_resized = cv2.resize(plot_img, (new_w, new_h))
                            
                            # Place di kanan bawah
                            y_offset = frame_h - new_h - 10
                            x_offset = frame_w - new_w - 10
                            
                            # Alpha blending untuk transparency
                            roi = frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
                            blended = cv2.addWeighted(roi, 0.3, plot_resized, 0.7, 0)
                            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = blended
                        
                        # Visualisasi ROIs
                        colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255)]
                        for i, poly in enumerate(all_polys):
                            cv2.polylines(frame, [poly], True, colors[i % len(colors)], 2)
                            
            else:
                # No face detected
                cv2.putText(frame, "No Face Detected", (frame.shape[1]//2 - 100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow('Enhanced rPPG System', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.use_pos_method = not self.use_pos_method
                print(f"Method changed to: {'POS' if self.use_pos_method else 'Green Channel'}")
            elif key == ord('r'):
                self.signal_buffer.clear()
                self.rgb_buffer.clear()
                self.bpm_history.clear()
                print("Buffers reset!")
        
        self.cap.release()
        cv2.destroyAllWindows()
        plt.close(self.fig)
        print("\nSystem stopped. Goodbye!")

if __name__ == "__main__":
    try:
        app = EnhancedRealTimeRPPG(buffer_size=150, fps=30)
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()