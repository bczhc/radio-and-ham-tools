#![feature(yeet_expr)]

use anyhow::anyhow;
use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use chrono::NaiveTime;
use clap::{Parser, ValueEnum};
use dasp::Sample;
use hound::{SampleFormat, WavSpec, WavWriter};
use lazy_regex::regex;
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::thread::spawn;
use std::{io, mem};
use crossbeam_channel::{bounded, Receiver, Sender};
use yeet_ops::yeet;

const SAMPLE_RATE: u64 = 768000_u64;
const CHANNEL_SIZE: usize = SAMPLE_RATE as usize * 10;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'c')]
    f_center: i64,
    /// Demodulation frequencies separated by comma. Format: <kHz>(LSB|USB); e.g.: 7056USB, 13200.2LSB
    #[arg(short = 'f')]
    f_target_list: String,
    #[arg(short = 'r', default_value = "6000")]
    output_sample_rate: u32,
    /// IQ raw input
    input: PathBuf,
    /// Wav output directory
    out_dir: PathBuf,
    #[arg(short = 'a', default_value = "1.0")]
    amplification: f64,
    #[arg(long = "swap-iq")]
    swap_iq: bool,
    /// Start time. Format: HH:mm:ss
    #[arg(long = "start", default_value = "00:00:00")]
    start: String,
    #[arg(long = "end")]
    /// End time. Format: HH:mm:ss
    end: Option<String>,
}

#[derive(ValueEnum, Debug, PartialEq, Eq, Clone, Copy)]
enum SsbType {
    Usb,
    Lsb,
}

impl SsbType {
    fn name(&self) -> &str {
        match self {
            SsbType::Usb => "USB",
            SsbType::Lsb => "LSB",
        }
    }
}

impl FromStr for SsbType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            _ if s.eq_ignore_ascii_case("usb") => Ok(Self::Usb),
            _ if s.eq_ignore_ascii_case("lsb") => Ok(Self::Lsb),
            _ => Err(anyhow!("Only USB and LSB are accepted")),
        }
    }
}

pub fn complex_hilbert(input: &[Complex64]) -> Vec<Complex<f64>> {
    let len = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);

    let mut fft_complex = input.iter().copied().collect::<Vec<_>>();
    fft.process(&mut fft_complex);

    let mut h_spectrum = vec![Complex::new(0.0, 0.0); len];

    if len % 2 == 0 {
        h_spectrum[0] = Complex::new(1.0, 0.0);
        h_spectrum[len / 2] = Complex::new(1.0, 0.0);
        for i in 1..(len / 2) {
            h_spectrum[i] = Complex::new(2.0, 0.0);
        }
    } else {
        h_spectrum[0] = Complex::new(1.0, 0.0);
        for i in 1..((len + 1) / 2) {
            h_spectrum[i] = Complex::new(2.0, 0.0);
        }
    }

    for i in 0..len {
        fft_complex[i] = fft_complex[i] * h_spectrum[i];
    }

    let mut ifft_complex = fft_complex.clone();
    let ifft = planner.plan_fft_inverse(len);
    ifft.process(&mut ifft_complex);

    for val in ifft_complex.iter_mut() {
        *val = *val / len as f64
    }

    ifft_complex
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let freq_list = parse_arg_frequencies(&args.f_target_list)?;
    println!("Frequency list: {:?}", freq_list);

    let f_center: i64 = args.f_center;
    let swap_iq = args.swap_iq;

    let zero_time = NaiveTime::from_hms_opt(0, 0, 0).unwrap();
    let start = NaiveTime::parse_from_str(&args.start, "%H:%M:%S")?;
    let skip_samples = (start - zero_time).num_seconds() as u64 * SAMPLE_RATE;
    let duration_samples = args
        .end
        .map(|x| {
            NaiveTime::parse_from_str(&x, "%H:%M:%S")
                .map(|x| (x - start).num_seconds() as u64 * SAMPLE_RATE)
        })
        .transpose()?;

    let mut task_vec = Vec::new();
    let mut feed_vec = Vec::new();
    for f in freq_list {
        let (tx, rx) = spawn_shift_and_downsample_thread(f_center, f.0, f.1)?;
        task_vec.push((f, rx));
        feed_vec.push(tx);
    }

    println!("Shifting spectrum and downsampling...");
    // spawn(move || {
    //     let mut feed_vec = feed_vec;
    //     let iq = read_iq_raw(args.input, swap_iq, skip_samples, duration_samples).unwrap();
    //     for (_, c) in iq {
    //         for x in &mut feed_vec {
    //             x.send(c).unwrap();
    //         }
    //     }
    // });
    for sender in feed_vec {
        let input = args.input.clone();
        spawn(move || {
            let iq = read_iq_raw(input, swap_iq, skip_samples, duration_samples).unwrap();
            for (_, c) in iq {
                sender.send(c).unwrap();
            }
        });
    }

    let mut result_vec = Vec::new();
    for x in task_vec {
        // wait for the result
        let result = x.1.recv().unwrap();
        result_vec.push((x.0, result));
    }
    result_vec.sort_by(|a, b| a.0 .0.cmp(&b.0 .0));
    println!("Performing Hilbert transform...");

    let result_vec = result_vec
        .par_iter()
        .map(|x| (x.0, complex_hilbert(&x.1)))
        .collect::<Vec<_>>();

    println!("Writing to wav files...");
    for (freq, samples) in result_vec {
        let filename = output_filename(freq.0, freq.1);
        let path = args.out_dir.join(filename);
        println!("Write to {}", path.display());
        let mut writer = WavWriter::new(
            BufWriter::new(File::create(path)?),
            WavSpec {
                channels: 1,
                sample_rate: 6000,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            },
        )?;
        for new_iq in samples {
            let pcm_sample = (new_iq.re * args.amplification).to_sample::<i16>();
            writer.write_sample(pcm_sample)?;
        }
    }

    Ok(())
}

fn read_iq_raw(
    wav_file: impl AsRef<Path> + Send + 'static,
    swap_iq: bool,
    samples_skip: u64,
    samples_duration: Option<u64>,
) -> anyhow::Result<Receiver<(u64, Complex64)>> {
    let (tx, rx) = bounded(CHANNEL_SIZE);
    spawn(move || {
        // Wav produced by SDR++ sometimes has a wrong header indicating a truncated length.
        // Read the wav on our own.
        let mut reader = BufReader::with_capacity(1048576, File::open(wav_file).unwrap());
        reader
            .seek(SeekFrom::Start(
                samples_skip * size_of::<f32>() as u64 * 2 /* stereo channel */,
            ))
            .unwrap();
        let mut iq_sample_n = 0_u64;
        loop {
            let s1 = reader.read_f32::<LE>();
            let s2 = reader.read_f32::<LE>();
            match (s1, s2) {
                (Err(e), _) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                (_, Err(e)) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                (Ok(mut i), Ok(mut q)) => {
                    if swap_iq {
                        mem::swap(&mut i, &mut q);
                    }
                    if iq_sample_n > samples_duration.unwrap_or(u64::MAX) {
                        break;
                    }
                    tx.send((iq_sample_n, Complex64::new(i as f64, q as f64)))
                        .unwrap();
                    iq_sample_n += 1;
                }
                _ => unreachable!(),
            }
        }
    });
    Ok(rx)
}

fn collect_f64le_iq_from_reader<R: Read>(mut reader: R) -> anyhow::Result<Vec<Complex64>> {
    let mut collected = Vec::new();
    loop {
        let s1 = reader.read_f64::<LE>();
        let s2 = reader.read_f64::<LE>();
        match (s1, s2) {
            (Err(e), _) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            (_, Err(e)) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            (Ok(i), Ok(q)) => {
                collected.push(Complex64::new(i, q));
            }
            _ => unreachable!(),
        }
    }
    Ok(collected)
}

fn parse_arg_frequencies(input: &str) -> anyhow::Result<Vec<(i64, SsbType)>> {
    let mut freq_list = Vec::new();
    let format = regex!(r"^([0-9\.]+)(lsb|usb)$");
    for freq in input.split(",") {
        let freq = freq.trim().to_ascii_lowercase();
        if !format.is_match(&freq) {
            yeet!(anyhow!("Invalid frequency: {}", freq));
        }
        let groups = format.captures_iter(&freq).next().unwrap();
        let khz = groups.get(1).unwrap().as_str();
        let ssb_type = groups.get(2).unwrap().as_str().parse::<SsbType>()?;
        let hz = Decimal::from_str(khz)? * Decimal::ONE_THOUSAND;
        freq_list.push((hz.to_i64().unwrap(), ssb_type));
    }
    Ok(freq_list)
}

fn spawn_shift_and_downsample_thread(
    f_center: i64,
    freq: i64,
    ssb_type: SsbType,
) -> anyhow::Result<(Sender<Complex64>, Receiver<Vec<Complex64>>)> {
    let f_shift = f_center - freq;

    let (tx, rx) = bounded(CHANNEL_SIZE);
    let (result_tx, result_rx) = bounded(1);

    let program = shell_words::split(
        format!("ffmpeg -ar {SAMPLE_RATE} -ac 2 -f f64le -i pipe:0 -ac 2 -ar 6000 -f f64le -")
            .as_str(),
    )
    .unwrap();
    let mut command = Command::new(&program[0])
        .args(&program[1..])
        .stderr(Stdio::inherit())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut program_in = BufWriter::new(command.stdin.take().unwrap());
    let program_out = BufReader::new(command.stdout.take().unwrap());

    spawn(move || {
        for (iq_n, iq) in rx.iter().enumerate() {
            let t = iq_n as f64 / SAMPLE_RATE as f64;
            // multiply by e^j2πft to shift the spectrum
            let mut shifted_iq: Complex64 = Complex64::from_polar(1.0, 2.0 * PI * t * f_shift as f64) * iq;
            // convert LSB to USB using spectrum mirroring
            if ssb_type == SsbType::Lsb {
                mem::swap(&mut shifted_iq.re, &mut shifted_iq.im);
            }
            // after the shift, resample the iq to a lower sample rate to reduce the hilbert transform work
            program_in.write_f64::<LE>(shifted_iq.re).unwrap();
            program_in.write_f64::<LE>(shifted_iq.im).unwrap();
        }
        drop(program_in);

        let status = command.wait().unwrap();
        if !status.success() {
            panic!("ffmpeg exits with non-zero status: {}", status);
        }
    });

    spawn(move || {
        let collected = collect_f64le_iq_from_reader(program_out).unwrap();
        result_tx.send(collected).unwrap();
    });

    Ok((tx, result_rx))
}

fn output_filename(hz: i64, ssb_type: SsbType) -> String {
    let khz = Decimal::from(hz) / Decimal::ONE_THOUSAND;
    format!("{}{}.wav", khz, ssb_type.name())
}
