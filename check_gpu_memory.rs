use gpu_attestor::gpu::cuda_driver::{CudaContext, CudaDevice};

fn main() -> anyhow::Result<()> {
    // Initialize CUDA
    let device = CudaDevice::init()?;
    println!("Device: {}", device.name()?);
    
    // Get memory info
    let (free, total) = device.memory_info()?;
    println!("Total VRAM: {} GB", total as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Free VRAM: {} GB", free as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("Used VRAM: {} GB", (total - free) as f64 / (1024.0 * 1024.0 * 1024.0));
    
    Ok(())
}