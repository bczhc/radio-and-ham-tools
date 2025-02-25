#![feature(let_chains)]

use crate::sample_formats::SampleFormat;
use anyhow::anyhow;
use hound::{WavSpec, WavWriter};
use num_complex::{Complex, Complex64};
use std::fs::File;
use std::io;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::spawn;

pub const SYNC_CHANNEL_SIZE: usize = 1048576;

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
