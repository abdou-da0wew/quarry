//! VRAM monitoring using NVML.
//!
//! Provides real-time GPU memory monitoring for NVIDIA GPUs.

use std::fmt;
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum VramError {
    #[error("NVML initialization failed: {0}")]
    NvmlInit(String),

    #[error("Failed to get GPU device: {0}")]
    DeviceError(String),

    #[error("Failed to read VRAM: {0}")]
    ReadError(String),

    #[error("NVML not available")]
    NotAvailable,
}

/// VRAM usage information
#[derive(Debug, Clone, Copy)]
pub struct VramUsage {
    /// Total VRAM in bytes
    pub total_bytes: u64,
    /// Used VRAM in bytes
    pub used_bytes: u64,
    /// Free VRAM in bytes
    pub free_bytes: u64,
    /// Usage percentage (0-100)
    pub utilization_percent: f64,
}

impl VramUsage {
    /// Check if usage is within the specified ceiling.
    pub fn within_ceiling(&self, ceiling_bytes: u64) -> bool {
        self.used_bytes <= ceiling_bytes
    }

    /// Get available VRAM for inference.
    pub fn available_for_inference(&self, ceiling_bytes: u64) -> u64 {
        ceiling_bytes.saturating_sub(self.used_bytes)
    }
}

impl fmt::Display for VramUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2} MB / {:.2} MB ({:.1}%)",
            self.used_bytes as f64 / 1_000_000.0,
            self.total_bytes as f64 / 1_000_000.0,
            self.utilization_percent
        )
    }
}

/// VRAM monitor using NVML.
pub struct VramMonitor {
    nvml: Option<nvml_wrapper::Nvml>,
    device_index: u32,
}

impl VramMonitor {
    /// Create a new VRAM monitor.
    ///
    /// Will return Ok with disabled monitoring if NVML is not available.
    pub fn new() -> Result<Self, VramError> {
        Self::with_device_index(0)
    }

    /// Create a VRAM monitor for a specific GPU device.
    pub fn with_device_index(device_index: u32) -> Result<Self, VramError> {
        match nvml_wrapper::Nvml::init() {
            Ok(nvml) => {
                debug!("NVML initialized successfully");
                Ok(Self {
                    nvml: Some(nvml),
                    device_index,
                })
            }
            Err(e) => {
                warn!("NVML not available, VRAM monitoring disabled: {}", e);
                Ok(Self {
                    nvml: None,
                    device_index,
                })
            }
        }
    }

    /// Check if NVML is available.
    pub fn is_available(&self) -> bool {
        self.nvml.is_some()
    }

    /// Get current VRAM usage.
    pub fn get_usage(&self) -> Result<VramUsage, VramError> {
        let nvml = self.nvml.as_ref().ok_or(VramError::NotAvailable)?;

        let device = nvml
            .device_by_index(self.device_index)
            .map_err(|e| VramError::DeviceError(e.to_string()))?;

        let mem_info = device
            .memory_info()
            .map_err(|e| VramError::ReadError(e.to_string()))?;

        let utilization_percent = if mem_info.total > 0 {
            (mem_info.used as f64 / mem_info.total as f64) * 100.0
        } else {
            0.0
        };

        Ok(VramUsage {
            total_bytes: mem_info.total,
            used_bytes: mem_info.used,
            free_bytes: mem_info.free,
            utilization_percent,
        })
    }

    /// Log current VRAM usage.
    pub fn log_usage(&self, context: &str) {
        match self.get_usage() {
            Ok(usage) => {
                info!(
                    context = context,
                    vram_used_mb = usage.used_bytes as f64 / 1_000_000.0,
                    vram_total_mb = usage.total_bytes as f64 / 1_000_000.0,
                    vram_utilization = usage.utilization_percent,
                    "VRAM usage: {}",
                    usage
                );
            }
            Err(e) => {
                debug!("Could not read VRAM: {}", e);
            }
        }
    }

    /// Estimate VRAM needed for a batch.
    ///
    /// # Arguments
    /// * `batch_size` - Number of sequences
    /// * `seq_len` - Sequence length
    /// * `hidden_dim` - Hidden dimension
    ///
    /// # Returns
    /// Estimated VRAM in bytes
    pub fn estimate_batch_vram(batch_size: usize, seq_len: usize, hidden_dim: usize) -> u64 {
        // Input tensors: input_ids + attention_mask (i64 each)
        let input_size = 2 * batch_size * seq_len * std::mem::size_of::<i64>();

        // Hidden states: batch_size * seq_len * hidden_dim * f32
        let hidden_size = batch_size * seq_len * hidden_dim * std::mem::size_of::<f32>();

        // Intermediate activations (estimate ~4x for transformer layers)
        let activations = hidden_size * 4;

        // Output embeddings: batch_size * embedding_dim * f32
        let output_size = batch_size * hidden_dim * std::mem::size_of::<f32>();

        // Overhead factor (ONNX Runtime overhead, CUDA context, etc.)
        let overhead = 50_000_000; // 50 MB

        (input_size + hidden_size + activations + output_size + overhead) as u64
    }

    /// Check if a batch will fit within VRAM ceiling.
    pub fn can_allocate(
        &self,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        ceiling_bytes: u64,
    ) -> Result<bool, VramError> {
        let estimated = Self::estimate_batch_vram(batch_size, seq_len, hidden_dim);

        match self.get_usage() {
            Ok(usage) => {
                let available = usage.available_for_inference(ceiling_bytes);
                debug!(
                    estimated_mb = estimated as f64 / 1_000_000.0,
                    available_mb = available as f64 / 1_000_000.0,
                    "Checking batch allocation"
                );
                Ok(estimated <= available)
            }
            Err(VramError::NotAvailable) => {
                // If NVML is not available, assume it will fit
                warn!("NVML not available, assuming batch will fit");
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }

    /// Find the maximum batch size that fits within VRAM.
    ///
    /// Starts from `max_batch_size` and halves until it fits.
    pub fn find_safe_batch_size(
        &self,
        max_batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        ceiling_bytes: u64,
    ) -> Result<usize, VramError> {
        let mut batch_size = max_batch_size;

        while batch_size > 0 {
            if self.can_allocate(batch_size, seq_len, hidden_dim, ceiling_bytes)? {
                return Ok(batch_size);
            }
            batch_size /= 2;
        }

        // Even batch_size=1 doesn't fit
        Err(VramError::ReadError(
            "Insufficient VRAM even for batch_size=1".to_string(),
        ))
    }
}

impl Default for VramMonitor {
    fn default() -> Self {
        Self::new().unwrap_or(Self { nvml: None, device_index: 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_estimate() {
        let estimate = VramMonitor::estimate_batch_vram(8, 256, 384);
        // Should be reasonable for MX250
        println!("Estimated VRAM for batch 8: {} MB", estimate as f64 / 1_000_000.0);
        assert!(estimate > 0);
    }

    #[test]
    fn test_vram_usage_display() {
        let usage = VramUsage {
            total_bytes: 2_000_000_000,
            used_bytes: 1_000_000_000,
            free_bytes: 1_000_000_000,
            utilization_percent: 50.0,
        };
        let s = format!("{}", usage);
        assert!(s.contains("1000.00 MB"));
    }

    #[test]
    fn test_within_ceiling() {
        let usage = VramUsage {
            total_bytes: 2_000_000_000,
            used_bytes: 1_000_000_000,
            free_bytes: 1_000_000_000,
            utilization_percent: 50.0,
        };
        assert!(usage.within_ceiling(1_500_000_000));
        assert!(!usage.within_ceiling(500_000_000));
    }
}
