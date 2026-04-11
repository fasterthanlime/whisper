unsafe extern "C" {
    fn mlx_set_cache_limit(res: *mut usize, limit: usize) -> std::ffi::c_int;
    fn mlx_clear_cache() -> std::ffi::c_int;
}

/// Install the mlx-rs error handler, replacing the default C handler
/// which calls `exit(255)` on any error.
fn ensure_mlx_error_handler() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        mlx_rs::error::setup_mlx_error_handler();
    });
}

/// Release unused MLX Metal buffers from the pool back to the system.
/// Safe to call concurrently — only frees buffers with no live references.
pub fn clear_mlx_cache() {
    ensure_mlx_error_handler();
    unsafe {
        mlx_clear_cache();
    }
}

/// Set the MLX Metal buffer cache limit. Buffers beyond this are returned
/// to the system instead of being pooled for reuse.
pub fn set_mlx_cache_limit(limit: usize) -> Result<usize, String> {
    ensure_mlx_error_handler();
    let mut prev = 0usize;
    let rc = unsafe { mlx_set_cache_limit(&mut prev, limit) };
    if rc != 0 {
        Err(format!("mlx_set_cache_limit failed (rc={rc})"))
    } else {
        Ok(prev)
    }
}
