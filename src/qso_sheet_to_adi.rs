#![feature(decl_macro)]
#![feature(yeet_expr)]

use anyhow::anyhow;
use calamine::{open_workbook, Data, DataType, Ods, Range, Reader};
use clap::Parser;
use hambands::band::types::Hertz;
use hambands::search::get_band_for_frequency;
use lazy_regex::regex;
use regex::Regex;
use rust_decimal::prelude::{FromPrimitive, Signed, ToPrimitive};
use rust_decimal::{Decimal, RoundingStrategy};
use std::fs::File;
use std::io;
use std::io::Write;
use std::ops::{Deref, Div, Mul};
use std::path::PathBuf;
use std::str::FromStr;
use yeet_ops::yeet;

#[derive(Parser)]
struct Args {
    /// Document input
    input: PathBuf,
    /// ADIF output
    output: PathBuf,
    /// My callsign
    station_callsign: String,
}

/// (<adif_name>, <value>)
type AdifRow = Vec<(&'static str, String)>;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut adif = File::create(&args.output)?;

    let mut doc: Ods<_> = open_workbook(&args.input)?;
    let worksheets = doc.worksheets();
    for (_, sheet) in worksheets {
        process_sheet(&sheet, &args, &mut adif)?;
        println!();
    }
    
    Ok(())
}

fn process_sheet(sheet: &Range<Data>, args: &Args, mut out: impl Write) -> anyhow::Result<()>{
    let header = sheet.headers().unwrap();

    let mut row_counter = 0_u64;
    for r in sheet.rows().skip(1) {
        println!("{:?}", r[1].as_duration());
        let row = parse_sheet_row(r, &header, &args)?;
        write_adif_row(&row, &mut out)?;
        row_counter += 1;
    }

    println!("{row_counter} records have been written.");
    Ok(())
}

fn write_adif_row(row: &AdifRow, mut out: impl Write) -> io::Result<()> {
    for (key, value) in row {
        writeln!(out, "<{}:{}>{}", key, value.chars().count(), value)?;
    }
    writeln!(out, "<eor>")?;
    writeln!(out)?;
    Ok(())
}

fn parse_sheet_row(r: &[Data], header: &[String], args: &Args) -> anyhow::Result<AdifRow> {
    macro index($name:literal) {
        header.iter().position(|x| x == $name).unwrap()
    }

    let mut row = AdifRow::default();
    let date_string = r[index!("Date")].get_datetime_iso().unwrap();
    row.push(("qso_date", date_string.replace('-', "")));
    row.push((
        "time_on",
        duration_to_time_serial(&r[index!("Start (UTC)")].as_duration().unwrap()),
    ));

    let time_off = &r[index!("End (UTC)")].as_duration();
    if let Some(time_off) = time_off {
        row.push(("qso_date_off", date_string.replace('-', "")));
        row.push(("time_off", duration_to_time_serial(time_off)));
    }
    row.push(("call", r[index!("Call")].to_string()));

    let freq = parse_freq_field(&r[index!("Freq")].to_string())?;

    row.push(("band", freq.band));
    if let Some(f) = freq.freq {
        row.push(("freq", f));
    }
    let freq_rx_field = r[index!("Freq (Rx)")].to_string();
    if !freq_rx_field.is_empty() {
        let freq_rx = parse_freq_field(&freq_rx_field)?;
        row.push(("band_rx", freq_rx.band));
        if let Some(f) = freq_rx.freq {
            row.push(("freq_rx", f));
        }
    }

    let mut mode = r[index!("Mode")].to_string();
    if let "USB" | "LSB" = mode.as_str() {
        mode = "SSB".into();
    }
    row.push(("mode", mode));
    row.push(("rst_sent", r[index!("S/RST")].to_string()));
    row.push(("rst_rcvd", r[index!("R/RST")].to_string()));

    let tx_pwr_string = r[index!("S/PWR")].to_string();
    let tx_pwr_num = capture_first(regex!("^([0-9]+) ?[wW]$"), &tx_pwr_string);
    if let Some(t) = tx_pwr_num {
        row.push(("tx_pwr", t.into()));
    }

    row.push(("comment", r[index!("Remarks")].to_string()));
    row.push(("my_gridsquare", r[index!("My Grid")].to_string()));
    row.push(("station_callsign", args.station_callsign.clone()));
    Ok(row)
}

struct FreqBand {
    freq: Option<String>,
    band: String,
}

fn parse_freq_field(freq_field: &str) -> anyhow::Result<FreqBand> {
    fn parse_freq(f: &str) -> anyhow::Result<Hertz> {
        if f.ends_with("kHz") {
            let khz = capture_first(regex!(r"^([0-9\.]+) ?kHz$"), f).unwrap();
            let hertz = Decimal::from_str(khz)?
                .mul(Decimal::ONE_THOUSAND)
                .to_u64()
                .unwrap();
            Ok(hertz)
        } else if f.ends_with("MHz") {
            let khz = capture_first(regex!(r"^([0-9\.]+) ?MHz$"), f).unwrap();
            let hertz = Decimal::from_str(khz)?
                .mul(Decimal::ONE_THOUSAND)
                .mul(Decimal::ONE_THOUSAND)
                .to_u64()
                .unwrap();
            Ok(hertz)
        } else {
            Err(anyhow!("Unknown frequency: {}", f))
        }
    }

    let mut band: String = Default::default();
    let mut freq: Option<String> = None;
    let band_regex = regex!("^([0-9]+) ?[mM]$");
    if band_regex.is_match(freq_field) {
        // indicates a band
        let band_num = capture_first(band_regex, freq_field).unwrap();
        band = format!("{band_num}M");
    } else {
        // treat as a frequency
        let hertz = parse_freq(freq_field)?;
        let mut mhz = Decimal::from_u64(hertz)
            .unwrap()
            .div(Decimal::ONE_THOUSAND)
            .div(Decimal::ONE_THOUSAND);
        let mhz = mhz.to_string();
        freq = Some(mhz);

        let inferred_band = get_band_for_frequency(hertz)?;
        band = inferred_band.name.to_uppercase();
    }
    Ok(FreqBand { freq, band })
}

fn duration_to_time_serial(d: &chrono::Duration) -> String {
    let sec = d.num_minutes();
    let h = sec / 60;
    let m = sec % 60;
    format!("{:02}{:02}", h, m)
}

fn capture_first<'a, 'b>(r: &'a Regex, haystack: &'b str) -> Option<&'b str> {
    Some(r.captures_iter(haystack).next()?.get(1)?.as_str())
}
