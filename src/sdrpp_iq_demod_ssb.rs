#![feature(yeet_expr)]

use anyhow::anyhow;
use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use clap::{Parser, ValueEnum};
use crossbeam_channel::{bounded, Receiver, Sender};
use dasp::Sample;
use hound::{SampleFormat, WavSpec, WavWriter};
use lazy_regex::regex;
use num_complex::{Complex, Complex64};
use once_cell::sync::Lazy;
use radio_and_ham_tools::iq_raw::{parse_sample_range, read_iq_raw};
use rayon::prelude::*;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rustfft::FftPlanner;
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::thread::spawn;
use std::{io, mem, thread};
use threadpool::ThreadPool;
use yeet_ops::yeet;

const IQ_SAMPLE_RATE: u64 = 768000_u64;
const VOICE_SAMPLE_RATE: u64 = 6000_u64;
const CHANNEL_SIZE: usize = IQ_SAMPLE_RATE as usize * 10;

static NUM_CPUS_STR: Lazy<&'static str> = Lazy::new(|| num_cpus::get().to_string().leak());

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'c')]
    f_center: i64,
    /// Demodulation frequencies separated by comma. Format: <kHz>(LSB|USB); e.g.: 7056USB, 13200.2LSB
    #[arg(short = 'f')]
    f_target_list: String,
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
    /// Parallel jobs number
    #[arg(short, long, default_value = *NUM_CPUS_STR)]
    jobs: usize,
}

#[derive(ValueEnum, Debug, Eq, Clone, Copy)]
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

impl PartialEq<Self> for SsbType {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl PartialOrd<Self> for SsbType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if *self == Self::Lsb {
            // LSB < USB
            Some(Ordering::Less)
        } else {
            // USB > LSB
            Some(Ordering::Greater)
        }
    }
}

impl Ord for SsbType {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
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

    let mut freq_list = parse_arg_frequencies(&args.f_target_list)?;
    freq_list.sort_by(|a, b| {
        if a.0 == b.0 {
            a.1.cmp(&b.1)
        } else {
            a.0.cmp(&b.0)
        }
    });
    freq_list.dedup();
    println!("Frequency list: {:#?}", freq_list);
    println!("Jobs: {}", args.jobs);

    let f_center: i64 = args.f_center;
    let swap_iq = args.swap_iq;

    let sample_range = parse_sample_range(
        Some(&args.start),
        args.end.as_ref().map(|x| x.as_ref()),
        IQ_SAMPLE_RATE,
    )?;

    let pool = ThreadPool::new(args.jobs);

    let mut wav_writer_threads = Vec::new();
    for f in freq_list {
        let filename = output_filename(f.0, f.1);
        let path = args.out_dir.join(filename);
        let mut wav_writer = WavWriter::new(
            BufWriter::new(File::create(&path)?),
            WavSpec {
                channels: 1,
                sample_rate: VOICE_SAMPLE_RATE as u32,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            },
        )?;

        let worker = spawn_demodulation_worker(f_center, f.0, f.1)?;
        // feed thread
        let input = args.input.clone();
        pool.execute(move || {
            let iq = read_iq_raw(input, swap_iq, sample_range.start, sample_range.length).unwrap();
            for (_, iq) in iq {
                worker.0.send(iq).unwrap();
            }
        });
        // receiving thread
        let t = thread::Builder::new()
            .name(format!("Worker: {:?}", f))
            .spawn(move || {
                for pcm_sample in worker.1 {
                    wav_writer
                        .write_sample((pcm_sample * args.amplification).to_sample::<i16>())
                        .unwrap();
                }
                println!("Output to {} finished", path.display());
            })?;
        wav_writer_threads.push(t);
    }

    pool.join();
    wav_writer_threads
        .into_iter()
        .for_each(|x| x.join().unwrap());
    Ok(())
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

/// - In: Raw IQ samples
/// - Out: Demodulated PCM samples (mono f64 @ [VOICE_SAMPLE_RATE])
fn spawn_demodulation_worker(
    f_center: i64,
    freq: i64,
    ssb_type: SsbType,
) -> anyhow::Result<(Sender<Complex64>, impl IntoIterator<Item = f64>)> {
    let f_shift = f_center - freq;

    let (tx, rx) = bounded(CHANNEL_SIZE);

    let program = shell_words::split(
        format!("ffmpeg -hide_banner -ar {IQ_SAMPLE_RATE} -ac 2 -f f64le -i pipe:0 -ac 2 -ar {VOICE_SAMPLE_RATE} -f f64le -")
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
            let t = iq_n as f64 / IQ_SAMPLE_RATE as f64;
            // multiply by e^j2πft to shift the spectrum
            let mut shifted_iq: Complex64 =
                Complex64::from_polar(1.0, 2.0 * PI * t * f_shift as f64) * iq;
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

    let (hilbert_tx, hilbert_rx) = spawn_interval_hilbert_worker()?;

    spawn(move || {
        let mut reader = program_out;
        loop {
            let s1 = reader.read_f64::<LE>();
            let s2 = reader.read_f64::<LE>();
            match (s1, s2) {
                (Err(e), _) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                (_, Err(e)) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                (Ok(i), Ok(q)) => {
                    hilbert_tx.send(Complex64::new(i, q)).unwrap();
                }
                _ => unreachable!(),
            }
        }
    });

    let map = hilbert_rx.into_iter().map(|x| x.re);
    Ok((tx, map))
}

fn spawn_interval_hilbert_worker() -> anyhow::Result<(Sender<Complex64>, Receiver<Complex64>)> {
    const GROUP_LENGTH: u64 = 1 * VOICE_SAMPLE_RATE; // 1s as a hilbert group
    let (tx, rx) = bounded(CHANNEL_SIZE);
    let (result_tx, result_rx) = bounded(CHANNEL_SIZE);

    spawn(move || {
        let mut buffer = Vec::new();
        for iq in rx {
            buffer.push(iq);
            if buffer.len() as u64 == GROUP_LENGTH {
                let transformed = complex_hilbert(&buffer);
                for sample in transformed {
                    result_tx.send(sample).unwrap();
                }
                buffer.clear();
            }
        }
        // Handle any remaining samples
        if !buffer.is_empty() {
            let transformed = complex_hilbert(&buffer);
            for sample in transformed {
                result_tx.send(sample).unwrap();
            }
        }
    });

    Ok((tx, result_rx))
}

fn output_filename(hz: i64, ssb_type: SsbType) -> String {
    let khz = Decimal::from(hz) / Decimal::ONE_THOUSAND;
    format!("{}{}.wav", khz, ssb_type.name())
}
