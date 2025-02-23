use byteorder::{ReadBytesExt, WriteBytesExt, LE};
use clap::{Parser, ValueEnum};
use num_complex::Complex64;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
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
    /// IQ wav input
    input: PathBuf,
    /// Wav output
    output: PathBuf,
    #[arg(short = 'a', default_value = "60.0")]
    amplification: f64,
    #[arg(long = "swap-iq")]
    swap_iq: bool,
}

#[derive(ValueEnum, Debug, PartialEq, Eq, Clone)]
enum SsbType {
    Usb,
    Lsb,
}

/// FIXME: DSB demodulation  ????????????????????????????? No.
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

    let mut command = Command::new("ffmpeg")
        .args([
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "768000",
            "-i",
            "pipe:0",
            "-ar",
            format!("{}", args.output_sample_rate).as_str(),
            args.output.to_str().unwrap(),
            "-y",
        ])
        .stderr(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stdin(Stdio::piped())
        .spawn()?;
    let ffmpeg_stdin = command.stdin.take().unwrap();
    let mut ffmpeg_writer = BufWriter::new(ffmpeg_stdin);

    let iq = read_iq(args.input, swap_iq)?;
    for (n, c) in iq {
        let t = n as f64 / SAMPLE_RATE as f64;
        // multiply by e^j2πft to shift the spectrum
        let shifted_iq = Complex64::from_polar(1.0, 2.0 * PI * t * f_shift as f64) * c;
        let new_sample = shifted_iq.re * amplification;
        // sample format: PCM f32le
        ffmpeg_writer.write_f32::<LE>(new_sample as f32)?;
    }

    drop(ffmpeg_writer);
    let status = command.wait()?;
    if !status.success() {
        panic!("ffmpeg exits with non-zero status: {}", status);
    }

    Ok(())
}

fn read_iq(
    wav_file: impl AsRef<Path> + Send + 'static,
    swap_iq: bool,
) -> anyhow::Result<Receiver<(u64, Complex64)>> {
    let (tx, rx) = sync_channel(CHANNEL_SIZE);
    spawn(move || {
        // Wav produced by SDR++ sometimes has a wrong header indicating a truncated length.
        // Read the wav on our own.
        const SDRPP_WAV_HEADER_SIZE: usize = 44;
        let mut reader = BufReader::with_capacity(1048576, File::open(wav_file).unwrap());
        // skip the header
        io::copy(
            &mut reader.by_ref().take(SDRPP_WAV_HEADER_SIZE as u64),
            &mut io::sink(),
        )
        .unwrap();
        let mut sample_n = 0_u64;
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
                    let i_f64 = sample_s16_to_f64(i);
                    let q_f64 = sample_s16_to_f64(q);
                    tx.send((sample_n, Complex64::new(i_f64, q_f64))).unwrap();
                    sample_n += 1;
                }
                _ => unreachable!(),
            }
        }
    });
    Ok(rx)
}

#[inline]
fn sample_s16_to_f64(x: i16) -> f64 {
    x as f64 / i16::MAX as f64
}
