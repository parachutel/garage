#!/usr/bin/env bash
# This script installs garage on Linux distributions.
#
# NOTICE: To keep consistency across this script, scripts/setup_macos.sh and
# docker/Dockerfile.base, if there's any changes applied to this file,
# specially regarding the installation of dependencies, apply those same
# changes to the mentioned files.

# Exit if any error occurs
set -e

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
  printf '%s\n' "Installer of garage for Linux."
  printf 'Usage: %s [--mjkey <arg>] [--(no-)modify-bashrc] ' "$0"
  printf '[-h|--help]\n'
  printf '\t%s\n' "--mjkey: Path of the MuJoCo key (no default)"
  printf '\t%s' "--modify-bashrc,--no-modify-bashrc: Set environment variables "
  printf '%s\n' "required by garage in .bashrc (off by default)"
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
print_error() {
  printf "\e[0;31m%s\e[0m" "${1}"
}

print_warning() {
  printf "\e[0;33m%s\e[0m" "${1}"
}

# Verify this script is running from the correct folder (root directory)
dir_err_txt="Please run this script only from the root of the garage \
repository, i.e. you should run it using the command \
\"bash scripts/setup_linux.sh\""
if [[ ! -f setup.py ]] && [[ ! $(grep -Fq "name='rlgarage'," setup.py) ]]; then
  _PRINT_HELP=yes die "${dir_err_txt}" 1
fi

# Verify there's a file in the mjkey path
test "$(file -b --mime-type ${_arg_mjkey})" == "text/plain" \
  || _PRINT_HELP=yes die \
  "The path ${_arg_mjkey} of the MuJoCo key is not valid." 1

# Make sure that we're under the garage directory
GARAGE_DIR="$(readlink -f $(dirname $0)/..)"
cd "${GARAGE_DIR}"

# File where environment variables are stored
BASH_RC="${HOME}/.bashrc"

# Install dependencies
# For installing garage: wget, bzip2, unzip, git
# For building glfw: cmake, xorg-dev
# Required for mujoco_py: libglew-dev, patchelf, libosmesa6-dev
# Required for OpenAI gym: libpq-dev, ffmpeg, libjpeg-dev, swig, libsdl2-dev
# Required for OpenAI baselines: libopenmpi-dev, openmpi-bin
echo "Installing garage dependencies"
echo "You will probably be asked for your sudo password"
sudo apt -y -q update
sudo apt install -y \
  wget \
  bzip2 \
  unzip \
  git \
  cmake \
  xorg-dev \
  libglew-dev \
  patchelf \
  libosmesa6-dev \
  libpq-dev \
  ffmpeg \
  libjpeg-dev \
  swig \
  libsdl2-dev \
  libopenmpi-dev \
  openmpi-bin

# Build GLFW because the Ubuntu 16.04 version is too old
# See https://github.com/glfw/glfw/issues/1004
sudo apt purge -y libglfw*
GLFW_DIR="$(mktemp -d)/glfw"
git clone https://github.com/glfw/glfw.git "${GLFW_DIR}"
cd "${GLFW_DIR}"
git checkout 0be4f3f75aebd9d24583ee86590a38e741db0904
mkdir glfw-build
cd glfw-build
sudo cmake -DBUILD_SHARED_LIBS=ON -DGLFW_BUILD_EXAMPLES=OFF \
  -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF ..
sudo make -j"$(nproc)"
sudo make install
cd "${GARAGE_DIR}"

# Leave a note in ~/.bashrc for the added environment variables
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo -e "\n# Added by the garage installer" >> "${BASH_RC}"
fi

# Set up MuJoCo 2.0 (for gym and dm_control)
if [[ ! -d "${HOME}/.mujoco/mujoco200_linux" ]]; then
  mkdir -p "${HOME}"/.mujoco
  MUJOCO_ZIP="$(mktemp -d)/mujoco.zip"
  wget https://www.roboti.us/download/mujoco200_linux.zip -O "${MUJOCO_ZIP}"
  unzip -u "${MUJOCO_ZIP}" -d "${HOME}"/.mujoco
  ln -s "${HOME}"/.mujoco/mujoco200_linux "${HOME}"/.mujoco/mujoco200
fi
# dm_control viewer requires MUJOCO_GL to be set to work
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo "export MUJOCO_GL=\"glfw\"" >> "${BASH_RC}"
fi
# Configure MuJoCo as a shared library
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin"
LD_LIB_ENV_VAR="LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200"
LD_LIB_ENV_VAR="${LD_LIB_ENV_VAR}/bin\""
if [[ "${_arg_modify_bashrc}" = on ]]; then
  echo "export ${LD_LIB_ENV_VAR}" >> "${BASH_RC}"
fi

# We need a MuJoCo key to import mujoco_py
if [[ ! -f "${HOME}/.mujoco/mjkey.txt" ]]; then
  cp "${_arg_mjkey}" "${HOME}/.mujoco/mjkey.txt"
fi

if [[ "${_arg_modify_bashrc}" != on ]]; then
  echo -e "\nRemember to execute the following commands before running garage:"
  echo "export ${LD_LIB_ENV_VAR}"
  echo "You may wish to edit your .bashrc to prepend these commands."
fi

echo -e "\ngarage pre-requisites are installed! To make the changes take " \
        "effect, open a new terminal or call 'source ~/.bashrc'" \
  | fold -s
