// Copyright lowRISC contributors (OpenTitan project).
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, ensure, Result};
use std::borrow::Borrow;
//use std::time::Duration;
use std::rc::Rc;

use crate::app::TransportWrapper;
use crate::io::gpio::{ClockNature, Edge, GpioMonitoring, MonitoringEvent};

// This structure makes it easier to monitor GPIOs and supports dumping a trace
// in the VCD format for further examination. In addition, for easier debugging
// in tests, this structure can automatically dump the trace on drop.
pub struct GpioMon<'a> {
    // Transport wrapper.
    transport: &'a TransportWrapper,
    // Monitor object.
    monitor: Rc<dyn GpioMonitoring>,
    // Transport pins names.
    pins: Vec<String>,
    // Whether to dump the trace on drop.
    dump_on_drop: bool,
    // Are we still monitoring?
    still_monitoring: bool,
    // Waves.
    waves: Waves,
}

pub struct Waves {
    // Human pin names.
    pin_names: Vec<String>,
    // Initial levels.
    initial_levels: Vec<bool>,
    // Initial timestamp.
    initial_timestamp: u64,
    // Final timestamp.
    final_timestamp: u64,
    // Clock resolution.
    resolution: u64,
    // Complete trace since the beginning.
    events: Vec<MonitoringEvent>,
}

impl<'a> GpioMon<'a> {
    // Start monitoring pins and return a monitoring object. `pins` is an array of pairs
    // (transport pin name, human pin name) where the human name can be
    // the empty string if the tranposrt pin name is enough.
    pub fn start(
        transport: &'a TransportWrapper,
        pins: &[(&str, &str)],
        dump_on_drop: bool,
    ) -> Result<Self> {
        let pin_names = pins
            .iter()
            .map(|(name, hname)| {
                if hname.is_empty() {
                    name.to_string()
                } else {
                    hname.to_string()
                }
            })
            .collect::<Vec<_>>();

        let monitor = transport.gpio_monitoring()?;
        let clock_nature = monitor.get_clock_nature()?;
        let ClockNature::Wallclock { resolution, .. } = clock_nature else {
            bail!("the transport GPIO monitor does not have reliable clock source");
        };
        let pins = pins
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        let gpio_pins = transport.gpio_pins(&pins)?;
        let gpios_pins = &gpio_pins.iter().map(Rc::borrow).collect::<Vec<_>>();
        let start = monitor.monitoring_start(gpios_pins)?;

        Ok(GpioMon {
            transport,
            monitor,
            pins,
            dump_on_drop,
            still_monitoring: true,
            waves: Waves::new(pin_names, start.initial_levels, start.timestamp, resolution),
        })
    }

    pub fn timestamp_to_ns(&self, timestamp: u64) -> u64 {
        self.waves.timestamp_to_ns(timestamp)
    }

    pub fn signal_index(&self, name: &str) -> Option<usize> {
        self.waves.signal_index(name)
    }

    pub fn initial_levels(&self) -> Vec<bool> {
        self.waves.initial_levels()
    }

    pub fn all_events(&self) -> impl Iterator<Item = &MonitoringEvent> {
        self.waves.all_events()
    }

    pub fn read(&mut self, continue_monitoring: bool) -> Result<Vec<MonitoringEvent>> {
        ensure!(
            self.still_monitoring,
            "cannot call read() on GpioMon after monitoring has stopped"
        );
        let gpio_pins = self.transport.gpio_pins(&self.pins)?;
        let gpios_pins = &gpio_pins.iter().map(Rc::borrow).collect::<Vec<_>>();
        let resp = self
            .monitor
            .monitoring_read(gpios_pins, continue_monitoring)?;
        self.still_monitoring = continue_monitoring;
        self.waves.add_events(&resp.events);
        self.waves.final_timestamp = resp.timestamp;
        Ok(resp.events)
    }

    pub fn dump_on_drop(&mut self, dump_on_drop: bool) {
        self.dump_on_drop = dump_on_drop
    }

    pub fn dump_vcd(&self) -> String {
        self.waves.dump_vcd()
    }
}

/// Consume a GpioMon and return a copy of the waves. If the monitor
/// is still active, this function will try to stop it (and return an
/// error if this is not possible).
impl TryFrom<GpioMon<'_>> for Waves {
    type Error = anyhow::Error;

    fn try_from(mut gpio_mon: GpioMon<'_>) -> Result<Waves> {
        if gpio_mon.still_monitoring {
            let _ = gpio_mon.read(false)?;
        }
        gpio_mon.dump_on_drop = false;
        let waves = std::mem::replace(
            &mut gpio_mon.waves,
            Waves::new(Vec::new(), Vec::new(), 0, 1),
        );
        Ok(waves)
    }
}

impl Waves {
    pub fn new(
        pin_names: Vec<String>,
        initial_levels: Vec<bool>,
        initial_timestamp: u64,
        resolution: u64,
    ) -> Waves {
        Waves {
            pin_names,
            initial_levels,
            initial_timestamp,
            final_timestamp: initial_timestamp,
            resolution,
            events: vec![],
        }
    }

    // Returns the index of the signal created.
    pub fn add_signal(&mut self, name: String, initial_level: bool) -> usize {
        self.pin_names.push(name);
        self.initial_levels.push(initial_level);
        self.pin_names.len() - 1
    }

    pub fn add_event(&mut self, event: MonitoringEvent) {
        self.events.push(event)
    }

    pub fn add_events(&mut self, event: &[MonitoringEvent]) {
        self.events.extend_from_slice(event)
    }

    pub fn signal_index(&self, name: &str) -> Option<usize> {
        self.pin_names.iter().position(|x| x == name)
    }

    pub fn initial_levels(&self) -> Vec<bool> {
        self.initial_levels.clone()
    }

    pub fn set_final_timestamp(&mut self, ts: u64) {
        self.final_timestamp = ts;
    }

    pub fn timestamp_to_ns(&self, timestamp: u64) -> u64 {
        (timestamp - self.initial_timestamp) * 1000000000u64 / self.resolution
    }

    pub fn ns_to_timestamp(&self, ns: u64) -> u64 {
        self.initial_timestamp + self.resolution * ns / 1000000000u64
    }

    pub fn all_events(&self) -> impl Iterator<Item = &MonitoringEvent> {
        self.events.iter()
    }

    pub fn sort_events(&mut self) {
        self.events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    // This function assumes that the events are sorted.
    pub fn dump_vcd(&self) -> String {
        const SYMBOLS: &[char] = &['!', '#', '$', '%', '&', '(', ')'];
        assert!(self.pin_names.len() < SYMBOLS.len());
        let mut vcd = String::new();
        vcd.push_str("$timescale 1ns $end\n");
        vcd.push_str("$scope module opentitanlib $end\n");
        for (i, name) in self.pin_names.iter().enumerate() {
            vcd.push_str(&format!("$var wire 1 {} {name} $end\n", SYMBOLS[i]));
        }
        vcd.push_str("$upscope $end\n");
        vcd.push_str("$enddefinitions $end\n");

        vcd.push_str("#0");
        for (i, lvl) in self.initial_levels.iter().enumerate() {
            vcd.push_str(&format!(" {}{}", if *lvl { 1 } else { 0 }, SYMBOLS[i]));
        }
        vcd.push('\n');

        for event in &self.events {
            let ns = self.timestamp_to_ns(event.timestamp);
            let val = event.edge == Edge::Rising;
            vcd.push_str(&format!(
                "#{ns} {}{}\n",
                val as u8, SYMBOLS[event.signal_index as usize]
            ));
        }
        // Make sure that the final timestamp is after the last event.
        let final_ts = std::cmp::max(self.events.last().unwrap().timestamp, self.final_timestamp);
        vcd.push_str(&format!("#{}", self.timestamp_to_ns(final_ts),));

        vcd
    }
}

impl Drop for GpioMon<'_> {
    fn drop(&mut self) {
        // Stop monitoring, ignore error.
        let error = if self.still_monitoring {
            self.read(false).is_err()
        } else {
            false
        };
        if self.dump_on_drop {
            if error {
                log::error!("an error occured when reading the monitoring events, some events have been lost");
            }
            log::info!(
                "====[ VCD dump ]====\n{}\n====[ end dump ]====",
                self.dump_vcd()
            );
        }
    }
}