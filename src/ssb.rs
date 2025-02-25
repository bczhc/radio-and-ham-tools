use bczhc_lib::complex::integral::{Integrate, Simpson38};
use hound::{SampleFormat, WavSpec};
use num_complex::Complex64;
use radio_and_ham_tools::{read_audio_samples, sample_interpolate, swap_iq, IqWriter};
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;

fn main() -> anyhow::Result<()> {
    let iq_sample_rate = 48000;
    let iq_output = "/home/bczhc/iq.wav";

    let mut iq_writer = hound::WavWriter::new(
        BufWriter::new(File::create(iq_output)?),
        WavSpec {
            channels: 2,
            sample_rate: iq_sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )?;

    let mut wav_output = hound::WavWriter::new(
        File::create("/home/bczhc/out.wav")?,
        WavSpec {
            channels: 1,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )?;

    let samples1 = read_audio_samples("/home/bczhc/3_bp2k.wav")?;
    let samples2 = read_audio_samples("/home/bczhc/4_bp2k.wav")?;
    let hilbert1 = hilbert_transform::hilbert(&samples1);
    let hilbert2 = hilbert_transform::hilbert(&samples2);
    let longer;
    let shorter;
    if hilbert1.len() > hilbert2.len() {
        longer = hilbert1;
        shorter = hilbert2;
    } else {
        longer = hilbert2;
        shorter = hilbert1;
    }
    let f_shift = 2000.0;
    for (i, _) in shorter.iter().enumerate() {
        let t = i as f64 / iq_sample_rate as f64;
        let a = Complex64::from_polar(1.0, 2.0 * PI * f_shift * t);
        let iq = longer[i] * 0.5 + swap_iq(shorter[i]) * 0.5;
        let iq = iq * a;
        iq_writer.write_iq(iq)?;
        let pcm = iq.re;
        wav_output.write_sample(pcm as f32)?;
    }

    /*// let hilbert_window_samples = 5000;
    for (i, &s) in samples.iter().enumerate().skip(1) {
        // let d_t = hilbert_window_samples as f64 / sample_rate as f64;
        // let t_start = (i as u32 / hilbert_window_samples) as f64 * d_t;
        // let t_end = t_start + d_t;
        // let t = i as f64 / sample_rate as f64;
        // let mut hilbert = compute_hilbert_transform(&samples, t_start, t_end, sample_rate, t);
        // println!("{hilbert}");
        // println!("{:?}", (t_start, t_end));
        // if hilbert.is_nan() || hilbert.is_infinite() {
        //     hilbert = 0.0;
        // }
        // let expected = f64::sin(2.0 * PI * t * 500.0 * t);
        // println!("{}", hilbert - expected);
        // let iq = Complex64::new(samples[i], hilbert2[i].im);
        if i % 100 == 0 {
            println!("Write sample: {i}");
        }
        iq_writer.write_iq(hilbert2[i])?;
    }*/

    Ok(())
}

fn compute_hilbert_transform(
    samples: &[f64],
    t_start: f64,
    t_end: f64,
    sample_rate: u32,
    var_t: f64,
) -> f64 {
    Simpson38::complex_integral_rayon(2000 * 100, t_start, t_end, |tau| {
        if (var_t - tau).abs() < 0.00000001 {
            Complex64::new(0.0, 0.0)
        } else {
            // let v = sample_interpolate(samples, sample_rate, tau) / (var_t - tau);
            let v1 = sample_interpolate(samples, sample_rate, var_t - tau);
            let v2 = sample_interpolate(samples, sample_rate, var_t + tau);
            let v = (v1 - v2) / (2.0 * tau);
            Complex64::new(v, 0.0)
        }
    })
    .re * 2.0
        / PI
}
