use clap::Parser;
use num_complex::Complex64;
use radio_and_ham_tools::{create_sdrpp_wav_iq, IqWriter};
use rand::RngExt;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    image: PathBuf,
    iq_out: PathBuf,
    /// Seconds one image line will make
    #[arg(short = 'd', long, default_value = "0.01")]
    line_duration: f64,
    /// Inverse image colors
    #[arg(short, long, default_value = "false")]
    invert: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut image = image::open(args.image)?.flipv();
    if args.invert {
        image.invert();
    }
    let image = image.to_luma8();
    let (width, height) = image.dimensions();

    let sample_rate = 6000;
    let base_freq = 0.0; // 起始频率
    let spacing = 3000.0 / width as f64; // 每个像素间距 6Hz，300像素刚好 1800Hz，在 USB 带宽内
    let line_duration = args.line_duration; // 每一行持续 0.2 秒，让图拉长一点
    let num_samples = (sample_rate as f64 * line_duration) as usize;

    let mut iq_writer = create_sdrpp_wav_iq("/home/bczhc/iq2.wav", sample_rate)?;

    let mut rng = rand::rng();

    // 1. 预先给每个像素频道生成一个固定的随机初相位
    let mut initial_phases: Vec<f64> = (0..width)
        .map(|_| rng.random::<f64>() * 2.0 * std::f64::consts::PI)
        .collect();

    let mut global_n = 0f64;

    for y in 0..height {
        for _n in 0..num_samples {
            let t = global_n / sample_rate as f64;
            let mut sample = 0.0;

            for x in 0..width {
                let brightness = image.get_pixel(x, y).0[0] as f64 / 255.0;
                if brightness > 0.1 {
                    let freq = base_freq + (x as f64 * spacing);
                    // 2. 关键：加上那个固定的随机初相位 phi
                    let phi = initial_phases[x as usize];
                    sample += brightness * (2.0 * std::f64::consts::PI * freq * t + phi).sin();
                }
            }

            global_n += 1.0;
            iq_writer.write_iq_s16(Complex64::new(sample / width as f64, 0.0))?;
        }
    }

    Ok(())
}
