#!/bin/sh
# Compile the system audio helper into native/systemaudiotap.
# Needs the Swift toolchain that ships with Xcode or the Command Line Tools.
cd "$(dirname "$0")" || exit 1
swiftc -O -o systemaudiotap SystemAudioTap.swift \
    -framework CoreAudio -framework AudioToolbox -framework Foundation
echo "built native/systemaudiotap"
