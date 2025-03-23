#![feature(let_chains)]

use crate::sample_formats::SampleFormat;
use anyhow::anyhow;
use hound::{WavSpec, WavWriter};
use num_complex::{Complex, Complex64};
use rust_decimal::Decimal;
use rustfft::num_traits::{ToPrimitive, Zero};
use rustfft::FftPlanner;
use std::fs::File;
use std::io;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::ops::Deref;
use std::path::Path;
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::spawn;

pub const SYNC_CHANNEL_SIZE: usize = 10485760;

pub fn read_audio_samples(file: impl AsRef<Path>) -> anyhow::Result<Vec<f64>> {
    let audio_reader = hound::WavReader::new(File::open(file.as_ref())?)?;
    assert_eq!(audio_reader.spec().channels, 1);
    let samples = audio_reader
        .into_samples::<i16>()
        .map(|x| x.unwrap() as f64 / i16::MAX as f64)
        .collect::<Vec<_>>();
    Ok(samples)
}

pub fn sample_interpolate(samples: &[f64], sample_rate: u32, mut t: f64) -> f64 {
    if t > 1.0 {
        t = 1.0;
    }
    let mut sample_n = (sample_rate as f64 * t) as usize;
    if sample_n == sample_rate as usize {
        sample_n = sample_rate as usize - 1;
    }
    let i_t = t * sample_rate as f64 - sample_n as f64;
    let a = samples[sample_n];
    let &b = samples.get(sample_n + 1).unwrap_or(&a);
    linear_interpolate(a, b, i_t)
}

#[inline]
pub fn linear_interpolate(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

pub trait IqWriter {
    fn write_iq_f32(&mut self, c: Complex64) -> hound::Result<()>;
    fn write_iq_s16(&mut self, c: Complex64) -> hound::Result<()>;
}

impl<R> IqWriter for WavWriter<R>
where
    R: Write + Seek,
{
    fn write_iq_f32(&mut self, c: Complex64) -> hound::Result<()> {
        self.write_sample(c.re as f32)?;
        self.write_sample(c.im as f32)
    }

    fn write_iq_s16(&mut self, c: Complex64) -> hound::Result<()> {
        let i = (i16::MAX as f64 * c.re) as i16;
        let q = (i16::MAX as f64 * c.im) as i16;
        self.write_sample(i)?;
        self.write_sample(q)
    }
}

#[inline]
pub fn swap_iq(c: Complex64) -> Complex<f64> {
    Complex64::new(c.im, c.re)
}

pub fn ffmpeg_read_audio_pcm<F>(
    source: impl AsRef<Path>,
    sample_rate: u32,
    channel: u32,
) -> anyhow::Result<Receiver<F::SampleType>>
where
    F: SampleFormat + Send,
    F::SampleType: Send + 'static,
{
    let (tx, rx) = sync_channel(SYNC_CHANNEL_SIZE);
    let codec = F::FFMPEG_CODEC;
    let format = F::FFMPEG_FORMAT;
    let cmd = format!(
        "ffmpeg -v error -i {} -map 0:a -c:a {codec} -ac {channel} -ar {sample_rate} -f {format} -",
        shell_words::quote(source.as_ref().to_str().ok_or(anyhow!("Non UTF-8"))?)
    );
    let split = shell_words::split(&cmd).unwrap();
    let child = Command::new(&split[0])
        .args(&split[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    spawn(move || {
        let mut child = child;
        let stdout = child.stdout.take().unwrap();
        let mut stdout = BufReader::new(stdout);
        loop {
            let result = F::read_sample(&mut stdout);
            if let Err(e) = &result
                && e.kind() == io::ErrorKind::UnexpectedEof
            {
                break;
            }
            let sample = result.unwrap();
            tx.send(sample).unwrap();
        }
    });
    Ok(rx)
}

pub mod sample_formats {
    use byteorder::{ReadBytesExt, LE};
    use std::io;
    use std::io::Read;

    pub trait SampleFormat {
        const FFMPEG_FORMAT: &'static str;
        const FFMPEG_CODEC: &'static str;
        type SampleType;

        fn read_sample<R: Read>(reader: R) -> io::Result<Self::SampleType>;
    }

    pub struct S16LE;
    pub struct F32LE;
    pub struct F64LE;

    impl SampleFormat for S16LE {
        const FFMPEG_FORMAT: &'static str = "s16le";
        const FFMPEG_CODEC: &'static str = "pcm_s16le";
        type SampleType = i16;

        fn read_sample<R: Read>(mut reader: R) -> io::Result<Self::SampleType> {
            reader.read_i16::<LE>()
        }
    }

    impl SampleFormat for F32LE {
        const FFMPEG_FORMAT: &'static str = "f32le";
        const FFMPEG_CODEC: &'static str = "pcm_f32le";
        type SampleType = f32;

        fn read_sample<R: Read>(mut reader: R) -> io::Result<Self::SampleType> {
            reader.read_f32::<LE>()
        }
    }

    impl SampleFormat for F64LE {
        const FFMPEG_FORMAT: &'static str = "f64le";
        const FFMPEG_CODEC: &'static str = "pcm_f64le";
        type SampleType = f64;

        fn read_sample<R: Read>(mut reader: R) -> io::Result<Self::SampleType> {
            reader.read_f64::<LE>()
        }
    }
}

pub fn create_sdrpp_wav_iq(
    path: impl AsRef<Path>,
    sample_rate: u32,
) -> anyhow::Result<WavWriter<impl Write + Seek>> {
    Ok(WavWriter::new(
        BufWriter::new(File::create(path)?),
        WavSpec {
            channels: 2,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?)
}

pub mod iq_raw {
    use crate::SYNC_CHANNEL_SIZE;
    use byteorder::{ReadBytesExt, LE};
    use chrono::NaiveTime;
    use crossbeam_channel::{bounded, Receiver};
    use num_complex::Complex64;
    use std::fs::File;
    use std::io::{BufReader, Seek, SeekFrom};
    use std::path::Path;
    use std::thread::spawn;
    use std::{io, mem};

    pub fn parse_sample_range(
        start: Option<&str>,
        end: Option<&str>,
        sample_rate: u64,
    ) -> anyhow::Result<SampleRange> {
        let zero_time = NaiveTime::from_hms_opt(0, 0, 0).unwrap();
        let start = NaiveTime::parse_from_str(&start.unwrap_or("0:0:0"), "%H:%M:%S")?;
        let skip_samples = (start - zero_time).num_seconds() as u64 * sample_rate;
        let duration_samples = end
            .map(|x| {
                NaiveTime::parse_from_str(&x, "%H:%M:%S")
                    .map(|x| (x - start).num_seconds() as u64 * sample_rate)
            })
            .transpose()?;
        Ok(SampleRange {
            start: skip_samples,
            length: duration_samples,
        })
    }

    #[derive(Debug, Copy, Clone)]
    pub struct SampleRange {
        pub start: u64,
        pub length: Option<u64>,
    }

    pub fn read_iq_raw(
        iq_raw_file: impl AsRef<Path> + Send + 'static,
        swap_iq: bool,
        samples_skip: u64,
        samples_duration: Option<u64>,
    ) -> anyhow::Result<Receiver<(u64, Complex64)>> {
        let (tx, rx) = bounded(SYNC_CHANNEL_SIZE);
        spawn(move || {
            let mut reader = BufReader::with_capacity(1048576, File::open(iq_raw_file).unwrap());
            reader
                .seek(SeekFrom::Start(
                    samples_skip * size_of::<f32>() as u64 * 2, /* stereo channel */
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
}

pub struct Hertz(pub i64);

impl Deref for Hertz {
    type Target = i64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hertz {
    pub fn from_khz_str(khz: &str) -> anyhow::Result<Self> {
        let hz = Decimal::from_str(khz)? * Decimal::ONE_THOUSAND;
        Ok(Self(hz.to_i64().ok_or(anyhow!("Out of range"))?))
    }
}

pub fn complex_hilbert(input: &[Complex64]) -> Vec<Complex64> {
    let mut fft_buffer = vec![Default::default(); input.len()];
    let mut fft_spec_buffer = vec![Default::default(); input.len()];
    let mut data = Vec::from(input);
    complex_hilbert_inplace(&mut data, &mut fft_buffer, &mut fft_spec_buffer);
    data
}

pub fn complex_hilbert_inplace(
    input: &mut [Complex64],
    fft_buffer: &mut [Complex64],
    fft_spec_buffer: &mut [Complex64],
) {
    debug_assert!(fft_buffer.len() >= input.len());
    debug_assert!(fft_spec_buffer.len() >= input.len());
    let len = input.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);

    fft.process_with_scratch(input, fft_buffer);

    let h_spectrum = fft_spec_buffer;
    h_spectrum.fill(Complex64::zero());
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

    input.iter_mut().zip(h_spectrum).for_each(|(i, h)| *i *= *h);

    let ifft = planner.plan_fft_inverse(len);
    ifft.process(input);

    input.iter_mut().for_each(|x| *x /= len as f64);
}

#[cfg(test)]
pub mod test {
    use crate::complex_hilbert;
    use num_complex::Complex64;

    #[test]
    fn hilbert() {
        let data = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let hilbert = complex_hilbert(&data);
        println!("{:?}", hilbert);
    }
}
