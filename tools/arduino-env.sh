# Keep every byte arduino-cli writes INSIDE Friday's project folder.
#
# By default arduino-cli scatters its core, its tool downloads and its
# sketchbook through ~/.arduino15 and ~/Arduino -- outside the project, which
# CLAUDE.md Part 1 puts off-limits. These three variables redirect all of it,
# so flashing the Arduino touches nothing but ~/friday and needs no system
# package: no apt, nothing in /etc, nothing owned by root.
#
#   source tools/arduino-env.sh
#   ./tools/arduino-cli board list
export ARDUINO_DIRECTORIES_DATA="$HOME/friday/tools/arduino/data"
export ARDUINO_DIRECTORIES_DOWNLOADS="$HOME/friday/tools/arduino/downloads"
export ARDUINO_DIRECTORIES_USER="$HOME/friday/tools/arduino/user"
