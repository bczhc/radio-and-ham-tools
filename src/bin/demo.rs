#![feature(file_buffered)]
#![feature(iter_map_windows)]

use bitvec::prelude::Msb0;
use bitvec::view::BitView;
use dasp::Sample;
use hound::{SampleFormat, WavSpec};
use num_complex::{Complex, Complex64};
use radio_and_ham_tools::{
    create_sdrpp_wav_iq, ffmpeg_read_audio_pcm_filter, sample_formats, IqWriter,
};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Read;
use rand::RngExt;

fn main() -> anyhow::Result<()> {
    let sample_rate = 500000;
    let path = "/home/bczhc/Music/奢香夫人.wav";
    let filter = "firequalizer=gain='if(between(f,300,5000),0,-inf)':zero_phase=on";
    let samples =
        ffmpeg_read_audio_pcm_filter::<sample_formats::F64LE>(path, sample_rate, 1, filter)?;

    let samples = samples
        .iter()
        // .skip(sample_rate as usize * 60)
        .take(sample_rate as usize * 5)
        .collect::<Vec<_>>();

    let mut iq_writer = create_sdrpp_wav_iq("/home/bczhc/iq.wav", sample_rate)?;
    // // FM
    // let k_f = 100_000.0;
    // let mut phase = 0.0;
    // for (i, &s) in samples.iter().enumerate() {
    //     let f_inst = k_f * s;
    //     phase += 2.0 * PI * f_inst / sample_rate as f64;
    //     phase %= 2.0 * PI;
    //     let iq = Complex64::from_polar(0.8, phase);
    //     iq_writer.write_iq_s16(iq)?;
    // }

    // PM
    // let k_p = 1.5;
    // for (i, &s) in samples.iter().enumerate() {
    //     let phase = k_p * s;
    //     let iq = Complex64::from_polar(0.8, phase);
    //     iq_writer.write_iq_s16(iq)?;
    // }

    let file = File::open_buffered("/home/bczhc/vsbm.html")?;
    let mut bytes = file.bytes().map(Result::unwrap).collect::<Vec<_>>();
    let bits = bytes.view_bits::<Msb0>();
    let mut bits = bits.into_iter();

    // QPSK - Return-To-Zero
    // let baud = 1000;
    // let samples_per_bit = sample_rate / baud;
    // let mut target_i = 0.0;
    // let mut target_q = 0.0;
    //
    // for idx in 0..sample_rate {
    //     if idx % samples_per_bit == 0 {
    //         let b1 = *bits.next().unwrap();
    //         let b2 = *bits.next().unwrap();
    //         target_i = if b1 { 1.0 } else { -1.0 };
    //         target_q = if b2 { 1.0 } else { -1.0 };
    //     }
    //     let sine_shape_multiplier =
    //         (((idx % samples_per_bit) as f64 / samples_per_bit as f64) * PI).sin();
    //     let sent_i = sine_shape_multiplier * target_i;
    //     let sent_q = sine_shape_multiplier * target_q;
    //
    //     iq_writer.write_iq_s16(Complex64::new(sent_i, sent_q))?;
    // }

    // BPSK - Non-Return-To-Zero
    // let baud = 500;
    // let samples_per_bit = sample_rate / baud;
    // let bits_i_data = bits.map(|x| if *x { -1.0 } else { 1.0 });
    // for group in bits_i_data.map_windows(|x: &[f64; 3]| *x) {
    //     let (prev, curr, next) = (group[0], group[1], group[2]);
    //
    //     for i in 0..(samples_per_bit / 2) {
    //         let progress = i as f64 / samples_per_bit as f64;
    //         let sent_i = match (prev, curr) {
    //             (1.0, -1.0) => -(PI * progress).sin(),
    //             (-1.0, 1.0) => (PI * progress).sin(),
    //             (-1.0, -1.0) => -1.0,
    //             (1.0, 1.0) => 1.0,
    //             _=> unreachable!()
    //         };
    //         iq_writer.write_iq_s16(Complex64::new(sent_i, 0.0))?;
    //     }
    //     for i in 0..(samples_per_bit / 2) {
    //         let progress = i as f64 / samples_per_bit as f64 + 0.5;
    //         let sent_i = match (curr, next) {
    //             (1.0, -1.0) => (PI * progress).sin(),
    //             (-1.0, 1.0) => -(PI * progress).sin(),
    //             (-1.0, -1.0) => -1.0,
    //             (1.0, 1.0) => 1.0,
    //             _=> unreachable!()
    //         };
    //         iq_writer.write_iq_s16(Complex64::new(sent_i, 0.0))?;
    //     }
    // }

    // AM, DSB-SC, USB
    // let mut iq_out = [(); 4].map(|_| Vec::new());
    //
    // let am_dc = 1.0;
    // for (i, &s) in samples.iter().enumerate() {
    //     let t = i as f64 / sample_rate as f64;
    //     // AM
    //     let am_s = (s + am_dc) / (1.0 + am_dc) * 0.8;
    //     let am_iq = Complex64::new(am_s, 0.0);
    //     iq_out[0].push(am_iq);
    //
    //     // DSB-SC
    //     let dsb_sc_iq = Complex64::new(s * 0.8, 0.0);
    //     iq_out[1].push(dsb_sc_iq);
    //
    //     // CW (OOK)
    //     let on = (t as u32) % 2;
    //     iq_out[3].push(Complex64::new(on as f64 * 0.8, 0.0));
    // }

    // USB
    // for c in samples.chunks(4096) {
    //     let hilbert = hilbert(c);
    //     hilbert
    //         .into_iter()
    //         .map(|x| x * 0.05)
    //         .for_each(|x| iq_out[2].push(x));
    // }

    // let mut iq_writer = create_sdrpp_wav_iq("/home/bczhc/iq.wav", sample_rate)?;
    //
    // for i in 0..samples.len() {
    //     let t = i as f64 / sample_rate as f64;
    //     iq_writer.write_iq_s16(
    //         (iq_out[0][i] * freq_shift_multiplier(10_000.0, t)
    //             + iq_out[1][i] * freq_shift_multiplier(25_000.0, t)
    //             + iq_out[2][i] * freq_shift_multiplier(35_000.0, t)
    //             + iq_out[3][i] * freq_shift_multiplier(17_000.0, t))
    //             / iq_out.len() as f64,
    //     )?;
    // }

    // 2-FSK
    // const BITS_COUNT: usize = 1024;
    // let mut bits = bits.take(BITS_COUNT);
    // let baud = 200;
    // let samples_per_bit = sample_rate / baud;
    // let freq_table = [-(baud as f64 / 2.0), baud as f64 / 2.0];
    //
    // let mut phase = 0.0;
    // let mut freq = 0.0;
    // for i in 0..(BITS_COUNT as u32 * samples_per_bit) {
    //     if i % samples_per_bit == 0 {
    //         let b = *bits.next().unwrap();
    //         freq = freq_table[b as usize];
    //     }
    //     let phase_step = 2.0 * PI / sample_rate as f64 * freq;
    //     phase += phase_step;
    //     let iq = Complex64::from_polar(0.8, phase);
    //     iq_writer.write_iq_s16(iq)?;
    // }

    let image = image::open("/home/bczhc/1.png")?;
    let image = image.to_luma8();
    let width = 300;
    let height = 300;
    println!("{}", image.get_pixel(0, 0).0[0]);

    let sample_rate = 6000u32;
    let fft_len = 600;
    let mut wav_writer = hound::WavWriter::new(
        File::create_buffered("/home/bczhc/out.wav")?,
        WavSpec {
            sample_rate,
            sample_format: SampleFormat::Int,
            channels: 1,
            bits_per_sample: 2 * 8,
        },
    )?;

    // Spectrum painting - my awful attempt
    // let c2r = realfft::RealFftPlanner::<f64>::new().plan_fft_inverse(fft_len);
    // let mut rng = rng();
    // for i in 0..10 {
    //     let mut spectrum = [(); 301].map(|_| Complex64::default());
    //     let mut out_samples = [0.0; 600];
    //
    //     for x in 0..300 {
    //         let span_avg = ((i * 30)..(i * 30 + 30))
    //             .map(|y| image.get_pixel(x, y).0[0] as f64 / 255.0)
    //             .sum::<f64>()
    //             / 30.0;
    //         let phi = 2.0 * PI * rng.random::<f64>();
    //         spectrum[x as usize + 1] = Complex64::from_polar(span_avg, phi);
    //     }
    //
    //     spectrum.last_mut().unwrap().im = 0.0;
    //     c2r.process(&mut spectrum, &mut out_samples)?;
    //     for x in out_samples {
    //         wav_writer.write_sample(x.to_sample::<i16>())?;
    //     }
    // }


    Ok(())
}

#[inline]
fn freq_shift_multiplier(f_shift: f64, t: f64) -> Complex<f64> {
    Complex64::from_polar(1.0, 2.0 * PI * f_shift * t)
}

#[allow(unused)]
fn unsee() -> Option<()> {
    panic!("I can't unsee it!")
}
