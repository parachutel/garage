#!/usr/bin/env bash
# This script installs garage on macOS distributions.
#
# NOTICE: To keep consistency across this script, scripts/setup_linux.sh and
# docker/Dockerfile.base, if there's any changes applied to this file,
# specially regarding the installation of dependencies, apply those same
# changes to the mentioned files.

# Exit if any error occurs
set -e

# Add macOS versions where garage is successfully installed in this list
VERIFIED_MACOS_VERSIONS=(
  "10.12"
  "10.13.6"
  "10.14"
  "10.14.4"
  "10.14.5"
)

### START OF CODE GENERATED BY Argbash v2.6.1 one line above ###
die()
{
  local _ret=$2
  test -n "$_ret" || _ret=1
  test "$_PRINT_HELP" = yes && print_help >&2
  echo "$1" >&2
  exit ${_ret}
}

begins_with_short_option()
{
  local first_option all_short_options
  all_short_options='h'
  first_option="${1:0:1}"
  test "$all_short_options" = "${all_short_options/$first_option/}" && \
    return 1 || return 0
}



# THE DEFAULTS INITIALIZATION - POSITIONALS
_positionals=()
# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_mjkey=
_arg_modify_bashrc="off"

print_help ()
{
  printf '%s\n' "Installer of garage for macOS."
  printf 'Usage: %s [--mjkey <arg>] [--(no-)modify-bashrc] ' "$0"
  printf '[-h|--help]\n'
  printf '\t%s\n' "--mjkey: Path of the MuJoCo key (no default)"
  printf '\t%s' "--modify-bashrc,--no-modify-bashrc: Set environment "
  printf '%s\n' "variables in .bash_profile (off by default)"
  printf '\t%s\n' "-h,--help: Prints help"
}

parse_commandline ()
{
  while test $# -gt 0
  do
    _key="$1"
    case "$_key" in
      --mjkey)
        test $# -lt 2 && \
          die "Missing value for the optional argument '$_key'." 1
        _arg_mjkey="$2"
        shift
        ;;
      --mjkey=*)
        _arg_mjkey="${_key##--mjkey=}"
        ;;
      --no-modify-bashrc|--modify-bashrc)
        _arg_modify_bashrc="on"
        test "${1:0:5}" = "--no-" && _arg_modify_bashrc="off"
        ;;
      -h|--help)
        print_help
        exit 0
        ;;
      -h*)
        print_help
        exit 0
        ;;
      *)
        _PRINT_HELP=yes die "FATAL ERROR: Got an unexpected argument '$1'" 1
        ;;
    esac
    shift
  done
}


parse_commandline "$@"
### END OF CODE GENERATED BY Argbash (sortof) ### ])

# Utility functions
script_dir_path() {
  SCRIPT_DIR="$(dirname ${0})"
  [[ "${SCRIPT_DIR}" = /* ]] && echo "${SCRIPT_DIR}" || \
    echo "${PWD}/${SCRIPT_DIR#./}"
}

# red text
print_error() {
  echo -e "\033[0;31m${@}\033[0m"
}

# yellow text
print_warning() {
  echo -e "\033[0;33m${@}\033[0m"
}

# Obtain the macOS version
VER="$(sw_vers -productVersion)"

if [[ ! " ${VERIFIED_MACOS_VERSIONS[@]} " =~ " ${VER} " ]]; then
  print_warning "You are attempting to install garage on a version of macOS" \
    "which we have not verified is working." | fold -s
  print_warning "\ngarage relies on community contributions to support macOS\n"
  print_warning "If this installation is successful, please add your macOS" \
    "version to VERIFIED_MACOS_VERSIONS to" \
    "https://github.com/rlworkgroup/garage/blob/master/scripts/setup_macos.sh" \
    "on GitHub and submit a pull request to rlworkgroup/garage to help out" \
    "future users. If the installation is not initially successful, but you" \
    "find changes which fix it, please help us out by submitting a PR with" \
    "your updates to the setup script." \
    | fold -s
  while [[ "${continue_var}" != "y" ]]; do
    read -p "Continue? (y/n): " continue_var
    if [[ "${continue_var}" = "n" ]]; then
      exit
    fi
  done
fi

# Verify this script is running from the correct folder (root directory)
dir_err_txt="Please run this script only from the root of the garage \
repository, i.e. you should run it using the command \
\"bash scripts/setup_macos.sh\""
if [[ ! -f setup.py ]] && [[ ! $(grep -Fq "name='rlgarage'," setup.py) ]]; then
  _PRINT_HELP=yes die "${dir_err_txt}" 1
fi

# Verify there's a file in the mjkey path
test "$(file -b --mime-type ${_arg_mjkey})" == "text/plain" \
  || _PRINT_HELP=yes die \
  "The path ${_arg_mjkey} of the MuJoCo key is not valid." 1

# Make sure that we're under the garage directory
GARAGE_DIR="$(dirname $(script_dir_path))"
cd "${GARAGE_DIR}"

# File where environment variables are stored
BASH_PROF="${HOME}/.bash_profile"

# Install dependencies
echo "Installing garage dependencies"

# Homebrew is required first to install the other dependencies
hash brew 2>/dev/null || {
  # Install the Xcode Command Line Tools
  set +e
  xcode-select --install
  set -e
  # Install Homebrew
  /usr/bin/ruby -e "$(curl -fsSL \
    https://raw.githubusercontent.com/Homebrew/install/master/install)"
}

# For installing garage: bzip2, git, glfw, unzip, wget
# For building glfw: cmake
# Required for OpenAI gym: cmake boost boost-python ffmpeg sdl2 swig wget
# Required for OpenAI baselines: cmake openmpi
brew update
set +e
brew install \
  gcc@8 \
  bzip2 \
  git \
  glfw \
  unzip \
  wget \
  cmake \
  boost \
  boost-python \
  ffmpeg \
  sdl2 \
  swig \
  openmpi
set -e

# Leave a note in ~/.bash_profile for the added environment variables
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo -e "\n# Added by the garage installer" >> "${BASH_PROF}"
fi

# Set up MuJoCo 2.0 (for gym and dm_control)
if [[ ! -d "${HOME}/.mujoco/mujoco200_macos" ]]; then
  mkdir -p "${HOME}"/.mujoco
  MUJOCO_ZIP="$(mktemp -d)/mujoco.zip"
  wget https://www.roboti.us/download/mujoco200_macos.zip -O "${MUJOCO_ZIP}"
  unzip -u "${MUJOCO_ZIP}" -d "${HOME}"/.mujoco
  ln -s "${HOME}"/.mujoco/mujoco200_macos "${HOME}"/.mujoco/mujoco200
fi
# dm_control viewer requires MUJOCO_GL to be set to work
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo "export MUJOCO_GL=\"glfw\"" >> "${BASH_PROF}"
fi
# Configure MuJoCo as a shared library
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin"
LD_LIB_ENV_VAR="LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200"
LD_LIB_ENV_VAR="${LD_LIB_ENV_VAR}/bin\""
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo "export ${LD_LIB_ENV_VAR}" >> "${BASH_PROF}"
fi

# We need a MuJoCo key to import mujoco_py
if [[ ! -f "${HOME}/.mujoco/mjkey.txt" ]]; then
  cp "${_arg_mjkey}" "${HOME}/.mujoco/mjkey.txt"
fi

# Add garage to python modules
if [[ "${_arg_modify_bashrc}" != on ]]; then
  echo -e "\nRemember to execute the following commands before running garage:"
  echo "${LD_LIB_ENV_VAR}"
  echo "You may wish to edit your .bash_profile to prepend these commands."
fi

echo -e "\ngarage pre-requisites are installed! To make the changes take " \
        "effect, open a new terminal or call 'source ~/.bash_profile'" \
  | fold -s
