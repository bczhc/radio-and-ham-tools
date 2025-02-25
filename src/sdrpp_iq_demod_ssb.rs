use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use chrono::NaiveTime;
use clap::{Parser, ValueEnum};
use dasp::Sample;
use hound::{SampleFormat, WavSpec};
use num_complex::{Complex, Complex64};
use rustfft::FftPlanner;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::spawn;
use std::{io, mem};

const SAMPLE_RATE: u64 = 768000_u64;
const CHANNEL_SIZE: usize = SAMPLE_RATE as usize * 10;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short = 'c')]
    f_center: i64,
    #[arg(short = 'f')]
    f_target: i64,
    #[arg(short = 't', default_value = "usb")]
    ssb_type: SsbType,
    #[arg(short = 'r', default_value = "6000")]
    output_sample_rate: u32,
    /// IQ raw input
    input: PathBuf,
    /// Wav output
    output: PathBuf,
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

#[derive(ValueEnum, Debug, PartialEq, Eq, Clone)]
enum SsbType {
    Usb,
    Lsb,
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

    let f_center: i64 = args.f_center;
    let amplification = args.amplification;
    let f_target: i64 = args.f_target;
    let mut swap_iq = args.swap_iq;
    let mut f_shift = f_center - f_target;
    if args.ssb_type == SsbType::Lsb {
        f_shift = -f_shift;
        swap_iq = !swap_iq;
    }

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

    let program = shell_words::split(
        format!("ffmpeg -ar {SAMPLE_RATE} -ac 2 -f f64le -i pipe:0 -ac 2 -ar 6000 -f f64le -")
            .as_str(),
    )?;
    let mut command = Command::new(&program[0])
        .args(&program[1..])
        .stderr(Stdio::inherit())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    let mut program_in = BufWriter::new(command.stdin.take().unwrap());
    spawn(move || {
        let iq = read_iq_raw(args.input, swap_iq, skip_samples, duration_samples).unwrap();
        for (n, c) in iq {
            let t = n as f64 / SAMPLE_RATE as f64;
            // multiply by e^j2πft to shift the spectrum
            let shifted_iq = Complex64::from_polar(1.0, 2.0 * PI * t * f_shift as f64) * c;
            // after the shift, resample the iq to a lower sample rate to reduce the hilbert transform work
            program_in.write_f64::<LE>(shifted_iq.re).unwrap();
            program_in.write_f64::<LE>(shifted_iq.im).unwrap();
        }
    });

    let program_out = BufReader::new(command.stdout.take().unwrap());
    let new_samples = collect_f64le_iq_from_reader(program_out)?;
    let status = command.wait()?;
    if !status.success() {
        panic!("ffmpeg exits with non-zero status: {}", status);
    }

    let new_samples_hilbert = complex_hilbert(&new_samples);
    let mut out = hound::WavWriter::new(
        BufWriter::new(File::create(args.output)?),
        WavSpec {
            channels: 1,
            sample_rate: 6000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    )?;

    for s in new_samples_hilbert {
        let s = (s.re * amplification).to_sample::<f32>();
        out.write_sample(s)?
    }

    Ok(())
}

fn read_iq_raw(
    wav_file: impl AsRef<Path> + Send + 'static,
    swap_iq: bool,
    samples_skip: u64,
    samples_duration: Option<u64>,
) -> anyhow::Result<Receiver<(u64, Complex64)>> {
    let (tx, rx) = sync_channel(CHANNEL_SIZE);
    spawn(move || {
        // Wav produced by SDR++ sometimes has a wrong header indicating a truncated length.
        // Read the wav on our own.
        let mut reader = BufReader::with_capacity(1048576, File::open(wav_file).unwrap());
        reader
            .seek(SeekFrom::Start(
                samples_skip * 2 /* s16 size */ * 2, /* channel is stereo */
            ))
            .unwrap();
        let mut iq_sample_n = 0_u64;
        loop {
            let s1 = reader.read_i16::<LE>();
            let s2 = reader.read_i16::<LE>();
            match (s1, s2) {
                (Err(e), _) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                (_, Err(e)) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                (Ok(mut i), Ok(mut q)) => {
                    if swap_iq {
                        mem::swap(&mut i, &mut q);
                    }
                    let i_f64 = i.to_sample::<f64>();
                    let q_f64 = q.to_sample::<f64>();
                    if iq_sample_n > samples_duration.unwrap_or(u64::MAX) {
                        break;
                    }
                    tx.send((iq_sample_n, Complex64::new(i_f64, q_f64)))
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
