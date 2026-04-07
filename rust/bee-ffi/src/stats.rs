#[repr(C)]
#[derive(Clone, Copy)]
pub struct AsrEngineStats {
    pub cpu_percent: c_float,
    pub gpu_percent: c_float,
    pub vram_used_mb: c_float,
    pub ram_used_mb: c_float,
}

impl Default for AsrEngineStats {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            gpu_percent: 0.0,
            vram_used_mb: 0.0,
            ram_used_mb: 0.0,
        }
    }
}

struct StatsSampler {
    latest: Arc<Mutex<AsrEngineStats>>,
}

impl StatsSampler {
    // EMA smoothing factor: α=0.25 at 400ms → time constant ~1.4s
    const ALPHA: f32 = 0.25;

    fn new() -> Self {
        let latest = Arc::new(Mutex::new(AsrEngineStats::default()));
        let shared = Arc::clone(&latest);
        std::thread::Builder::new()
            .name("bee-stats".into())
            .spawn(move || {
                let mut last_cpu_us: u64 = 0;
                let mut last_wall = Instant::now();
                let mut smooth = AsrEngineStats::default();

                loop {
                    std::thread::sleep(Duration::from_millis(400));
                    let now = Instant::now();
                    let wall_us = now.duration_since(last_wall).as_micros() as u64;
                    last_wall = now;

                    let cpu_us = process_cpu_us();
                    let cpu_raw = if wall_us > 0 && last_cpu_us > 0 {
                        let delta = cpu_us.saturating_sub(last_cpu_us);
                        ((delta as f32 / wall_us as f32) * 100.0).min(100.0)
                    } else {
                        0.0
                    };
                    last_cpu_us = cpu_us;

                    let ram_raw = process_ram_mb();
                    let (gpu_raw, vram_raw) = sample_gpu_iokit().unwrap_or((0.0, 0.0));

                    let a = Self::ALPHA;
                    smooth.cpu_percent = a * cpu_raw + (1.0 - a) * smooth.cpu_percent;
                    smooth.gpu_percent = a * gpu_raw + (1.0 - a) * smooth.gpu_percent;
                    smooth.vram_used_mb = a * vram_raw + (1.0 - a) * smooth.vram_used_mb;
                    smooth.ram_used_mb = a * ram_raw + (1.0 - a) * smooth.ram_used_mb;

                    if let Ok(mut s) = shared.lock() {
                        *s = smooth;
                    }
                }
            })
            .expect("failed to spawn stats thread");
        Self { latest }
    }

    fn get(&self) -> AsrEngineStats {
        self.latest.lock().map(|s| *s).unwrap_or_default()
    }
}

fn process_cpu_us() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        let user = usage.ru_utime.tv_sec as u64 * 1_000_000 + usage.ru_utime.tv_usec as u64;
        let sys = usage.ru_stime.tv_sec as u64 * 1_000_000 + usage.ru_stime.tv_usec as u64;
        user + sys
    }
}

fn process_ram_mb() -> f32 {
    // proc_pidinfo(PROC_PIDTASKINFO) gives current resident set size.
    #[repr(C)]
    struct ProcTaskinfo {
        pti_virtual_size: u64,
        pti_resident_size: u64,
        pti_total_user: u64,
        pti_total_system: u64,
        pti_threads_user: u64,
        pti_threads_system: u64,
        pti_policy: i32,
        pti_faults: i32,
        pti_pageins: i32,
        pti_cow_faults: i32,
        pti_messages_sent: i32,
        pti_messages_received: i32,
        pti_syscalls_mach: i32,
        pti_syscalls_unix: i32,
        pti_csw: i32,
        pti_threadnum: i32,
        pti_numrunning: i32,
        pti_priority: i32,
    }
    extern "C" {
        fn proc_pidinfo(
            pid: i32,
            flavor: i32,
            arg: u64,
            buffer: *mut std::ffi::c_void,
            buffersize: i32,
        ) -> i32;
    }
    const PROC_PIDTASKINFO: i32 = 4;
    unsafe {
        let mut info: ProcTaskinfo = std::mem::zeroed();
        let ret = proc_pidinfo(
            libc::getpid(),
            PROC_PIDTASKINFO,
            0,
            &mut info as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<ProcTaskinfo>() as i32,
        );
        if ret > 0 {
            info.pti_resident_size as f32 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

/// IOKit / CoreFoundation bindings — GPU stats without spawning a subprocess.
#[allow(non_upper_case_globals, non_snake_case)]
mod iokit {
    use std::ffi::{c_char, c_void};
    pub(crate) type IoServiceT = u32;
    pub(crate) const IO_OBJECT_NULL: IoServiceT = 0;
    pub(crate) const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    pub(crate) const K_CF_NUMBER_SINT64_TYPE: i32 = 4;

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        pub(crate) fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        pub(crate) fn IOServiceGetMatchingService(
            masterPort: IoServiceT,
            matching: *mut c_void,
        ) -> IoServiceT;
        pub(crate) fn IORegistryEntryCreateCFProperties(
            entry: IoServiceT,
            properties: *mut *mut c_void,
            allocator: *const c_void,
            options: u32,
        ) -> i32;
        pub(crate) fn IOObjectRelease(object: IoServiceT) -> i32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub(crate) fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const c_char,
            encoding: u32,
        ) -> *mut c_void;
        pub(crate) fn CFDictionaryGetValue(
            the_dict: *const c_void,
            key: *const c_void,
        ) -> *const c_void;
        pub(crate) fn CFNumberGetValue(
            number: *const c_void,
            the_type: i32,
            value_ptr: *mut c_void,
        ) -> bool;
        pub(crate) fn CFRelease(cf: *const c_void);
    }
}

fn cf_str(s: &str) -> *mut std::ffi::c_void {
    use iokit::*;
    let cs = std::ffi::CString::new(s).unwrap();
    unsafe { CFStringCreateWithCString(std::ptr::null(), cs.as_ptr(), K_CF_STRING_ENCODING_UTF8) }
}

fn cf_dict_i64(dict: *const std::ffi::c_void, key: &str) -> Option<i64> {
    use iokit::*;
    let k = cf_str(key);
    if k.is_null() {
        return None;
    }
    let val = unsafe { CFDictionaryGetValue(dict, k) };
    unsafe { CFRelease(k) };
    if val.is_null() {
        return None;
    }
    let mut out: i64 = 0;
    unsafe {
        CFNumberGetValue(
            val,
            K_CF_NUMBER_SINT64_TYPE,
            &mut out as *mut _ as *mut std::ffi::c_void,
        );
    }
    Some(out)
}

fn sample_gpu_iokit() -> Option<(f32, f32)> {
    use iokit::*;
    use std::ffi::CString;
    unsafe {
        let service_name = CString::new("AGXAccelerator").ok()?;
        let service = IOServiceGetMatchingService(0, IOServiceMatching(service_name.as_ptr()));
        if service == IO_OBJECT_NULL {
            return None;
        }
        let mut props: *mut std::ffi::c_void = std::ptr::null_mut();
        let kr = IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0);
        IOObjectRelease(service);
        if kr != 0 || props.is_null() {
            return None;
        }

        let perf_key = cf_str("PerformanceStatistics");
        let perf_dict = CFDictionaryGetValue(props, perf_key);
        CFRelease(perf_key);

        let result = if perf_dict.is_null() {
            None
        } else {
            let gpu = cf_dict_i64(perf_dict, "Device Utilization %").unwrap_or(0) as f32;
            let vram_bytes = cf_dict_i64(perf_dict, "In use system memory").unwrap_or(0);
            Some((gpu, vram_bytes as f32 / (1024.0 * 1024.0)))
        };

        CFRelease(props);
        result
    }
}
